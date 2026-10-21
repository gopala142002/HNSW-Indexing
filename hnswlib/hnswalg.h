#pragma once
#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <Eigen/Dense>
#include "profiling.h"
// tree structures for finding best starting points for level0 search
// #include "vtree.h"
#include "mtree.h"
#include "pctree.h"
#include "vpttree.h"
#include "finger_opt.h"
#include "kmeanstree.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

using std::vector;
using std::priority_queue;
using std::pair;
using std::unordered_set;
using std::unordered_map;
using std::unique_lock;
using std::mutex;
using std::runtime_error;
using std::min;
using std::max;
using std::numeric_limits;
using std::ofstream;
using std::ifstream;
using std::cout;
using std::endl;
using std::string;
using std::atomic;
using std::unique_ptr;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::defer_lock;
using std::streampos;
using std::ios;
using std::istream;
using std::ostream;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> 
{
    public:

    int finger_rank_ = 16;
    bool use_finger_ = false;
    size_t finger_warmup_updates_ = 4;
    FingerGlobalData finger_global_;
    std::vector<FingerNodeData> finger_data_;

    // VoronoiTree* vtree_ = nullptr;
    MTree* mtree_ = nullptr;
    PCTree* pctree_ = nullptr;
    VantagePointTree* vpt_ = nullptr;
    KMeansTree* kmeanstree_ = nullptr;

    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable atomic<size_t> cur_element_count{0};  
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable atomic<size_t> num_deleted_{0};  
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    void setFingerRank(int rank) 
    {
        if (rank <= 0) throw std::runtime_error("FINGER rank must be positive.");
        finger_rank_ = rank;
        finger_global_.ready = false;
    }

    int getFingerRank() const 
    {
        return finger_rank_;
    }

    void setFingerWarmup(size_t updates) 
    {
        if (updates == 0) 
        {
            throw std::runtime_error("FINGER warmup must be positive.");
        }
        finger_warmup_updates_ = updates;
    }

    size_t getFingerWarmup() const 
    {
        return finger_warmup_updates_;
    }

    void setUseFinger(bool enable, int rank = -1) 
    {
        use_finger_ = enable;
        if (rank > 0) finger_rank_ = rank;
        finger_global_.ready = false;
        if (enable && finger_data_.size() < max_elements_)
            finger_data_.resize(max_elements_);
    }

    void buildFingerProjections(int target_rank = -1) 
    {
        if (target_rank > 0) finger_rank_ = target_rank;
        if (!use_finger_ || cur_element_count == 0) return;

        const int dim = static_cast<int>(data_size_ / sizeof(float));

        if (finger_rank_ <= 0 || finger_rank_ > dim)
            throw std::runtime_error("Invalid FINGER rank.");

        std::mt19937 rng(1234567);
        std::vector<float> samples;
        size_t sample_count = 0;

        for (tableint c = 0; c < cur_element_count; ++c) 
        {
            const int m0 = getListCount(get_linklist0(c));
            if (m0 == 0) 
            {
                continue;
            }

            tableint* neighbors = get_linklist0(c) + 1;
            std::uniform_int_distribution<int> pick(0, m0 - 1);
            tableint d = neighbors[pick(rng)];

            const float* cv = reinterpret_cast<const float*>(getDataByInternalId(c));
            const float* dv = reinterpret_cast<const float*>(getDataByInternalId(d));

            float cc = 0.0f;
            float cd = 0.0f;
            for (int i = 0; i < dim; ++i) 
            {
                cc += cv[i] * cv[i];
                cd += cv[i] * dv[i];
            }

            float beta = cc > 1e-20f ? cd / cc : 0.0f;
            for (int i = 0; i < dim; ++i)
                samples.push_back(dv[i] - beta * cv[i]);
            ++sample_count;
        }

        if (sample_count == 0)
            throw std::runtime_error("FINGER: graph has no usable Level-0 edges.");

        Eigen::MatrixXf Dres(dim, static_cast<Eigen::Index>(sample_count));
        for (size_t j = 0; j < sample_count; ++j)
            for (int i = 0; i < dim; ++i)
                Dres(i, static_cast<Eigen::Index>(j)) = samples[j * dim + i];



        samples.clear();
        samples.shrink_to_fit();

        Eigen::BDCSVD<Eigen::MatrixXf> svd(Dres, Eigen::ComputeThinU);
        Eigen::MatrixXf U = svd.matrixU();

        if (U.cols() < finger_rank_)
            throw std::runtime_error("FINGER: insufficient rank for projection.");

        finger_global_.dim = dim;
        finger_global_.rank = finger_rank_;
        finger_global_.projection.resize(static_cast<size_t>(finger_rank_) * dim);

        for (int r = 0; r < finger_rank_; ++r)
            for (int i = 0; i < dim; ++i)
                finger_global_.projection[static_cast<size_t>(r) * dim + i] = U(i, r);

        finger_global_.point_projection.resize(static_cast<size_t>(cur_element_count) * finger_rank_);

        finger_global_.point_norm_sq.resize(cur_element_count);

        for (tableint id = 0; id < cur_element_count; ++id) 
        {
            const float* v = reinterpret_cast<const float*>(getDataByInternalId(id));
            finger_global_.point_norm_sq[id] = finger_norm_sq(v, dim);
            finger_project(v,finger_global_.projection.data(),dim,finger_rank_,&finger_global_.point_projection[static_cast<size_t>(id) * finger_rank_]);
        }

        std::vector<float> true_angles;
        std::vector<float> approx_angles;
        true_angles.reserve(cur_element_count);
        approx_angles.reserve(cur_element_count);

        for (tableint c = 0; c < cur_element_count; ++c) 
        {
            const int m0 = getListCount(get_linklist0(c));
            if (m0 < 2) continue;

            tableint* neighbors = get_linklist0(c) + 1;
            std::uniform_int_distribution<int> pick(0, m0 - 1);
            int a = pick(rng);
            int b = pick(rng);
            if (a == b) b = (b + 1) % m0;

            tableint d1 = neighbors[a];
            tableint d2 = neighbors[b];

            const float* cv = reinterpret_cast<const float*>(getDataByInternalId(c));
            const float* v1 = reinterpret_cast<const float*>(getDataByInternalId(d1));
            const float* v2 = reinterpret_cast<const float*>(getDataByInternalId(d2));

            float cc = finger_global_.point_norm_sq[c];
            float cd1 = 0.0f, cd2 = 0.0f;
            for (int i = 0; i < dim; ++i) 
            {
                cd1 += cv[i] * v1[i];
                cd2 += cv[i] * v2[i];
            }

            float beta1 = cc > 1e-20f ? cd1 / cc : 0.0f;
            float beta2 = cc > 1e-20f ? cd2 / cc : 0.0f;

            std::vector<float> r1(dim), r2(dim);
            for (int i = 0; i < dim; ++i) 
            {
                r1[i] = v1[i] - beta1 * cv[i];
                r2[i] = v2[i] - beta2 * cv[i];
            }

            const float* p1 = &finger_global_.point_projection[static_cast<size_t>(d1) * finger_rank_];
            const float* p2 = &finger_global_.point_projection[static_cast<size_t>(d2) * finger_rank_];
            const float* pc = &finger_global_.point_projection[static_cast<size_t>(c) * finger_rank_];

            std::vector<float> pr1(finger_rank_), pr2(finger_rank_);
            for (int r = 0; r < finger_rank_; ++r) 
            {
                pr1[r] = p1[r] - beta1 * pc[r];
                pr2[r] = p2[r] - beta2 * pc[r];
            }

            true_angles.push_back(finger_cosine(r1.data(), r2.data(), dim));
            approx_angles.push_back(finger_cosine(pr1.data(), pr2.data(), finger_rank_));
        }

        if (true_angles.empty())
            throw std::runtime_error("FINGER: insufficient neighboring pairs.");

        float sum_true = 0.0f;
        float sum_approx = 0.0f;
        for (size_t i = 0; i < true_angles.size(); ++i) 
        {
            sum_true += true_angles[i];
            sum_approx += approx_angles[i];
        }

        finger_global_.mu_true = sum_true / true_angles.size();
        finger_global_.mu_approx = sum_approx / approx_angles.size();

        float var_true = 0.0f;
        float var_approx = 0.0f;
        for (size_t i = 0; i < true_angles.size(); ++i) 
        {
            float a = true_angles[i] - finger_global_.mu_true;
            float b = approx_angles[i] - finger_global_.mu_approx;
            var_true += a * a;
            var_approx += b * b;
        }

        finger_global_.sigma_true = std::sqrt(var_true / true_angles.size());
        finger_global_.sigma_approx = std::sqrt(var_approx / approx_angles.size());

        if (finger_global_.sigma_true < 1e-12f) 
            finger_global_.sigma_true = 1.0f;

        if (finger_global_.sigma_approx < 1e-12f) 
            finger_global_.sigma_approx = 1.0f;

        float epsilon = 0.0f;
        for (size_t i = 0; i < true_angles.size(); ++i) 
        {
            float transformed = (approx_angles[i] - finger_global_.mu_approx) * finger_global_.sigma_true / finger_global_.sigma_approx +finger_global_.mu_true;
            epsilon += std::fabs(transformed - true_angles[i]);
        }


        finger_global_.epsilon = epsilon / true_angles.size();

        finger_data_.resize(max_elements_);

        for (tableint c = 0; c < cur_element_count; ++c) 
        {
            const int m0 = getListCount(get_linklist0(c));
            FingerNodeData& node = finger_data_[c];
            node.neighbor_coeffs.resize(static_cast<size_t>(m0) * finger_rank_);
            node.residual_norms.resize(m0);
            node.neighbor_center_coeffs.resize(m0);

            const float* cv = reinterpret_cast<const float*>(getDataByInternalId(c));
            float cc = finger_global_.point_norm_sq[c];
            const float* pc = &finger_global_.point_projection[static_cast<size_t>(c) * finger_rank_];
            tableint* neighbors = get_linklist0(c) + 1;

            for (int n = 0; n < m0; ++n) 
            {
                tableint d = neighbors[n];
                const float* dv = reinterpret_cast<const float*>(getDataByInternalId(d));
                float cd = 0.0f;
                for (int i = 0; i < dim; ++i) cd += cv[i] * dv[i];
                float beta = cc > 1e-20f ? cd / cc : 0.0f;

                float residual_norm_sq = 0.0f;
                for (int i = 0; i < dim; ++i) 
                {
                    float diff = dv[i] - beta * cv[i];
                    residual_norm_sq += diff * diff;
                }

                node.residual_norms[n] = residual_norm_sq;
                node.neighbor_center_coeffs[n] = beta;

                const float* pd = &finger_global_.point_projection[static_cast<size_t>(d) * finger_rank_];
                float* out = &node.neighbor_coeffs[static_cast<size_t>(n) * finger_rank_];
                for (int r = 0; r < finger_rank_; ++r)
                    out[r] = pd[r] - beta * pc[r];
            }
        }

        finger_global_.ready = true;
    }

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    mutable vector<mutex> label_op_locks_;

    mutex global;

    vector<mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    // for Tri-Scheme precomputation
    std::vector<dist_t> tri_edge_distances_;
    std::vector<std::vector<std::vector<dist_t>>> upper_tri_edge_distances_;
    bool tri_ready_ = false;

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    vector<int> element_levels_; 

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable mutex label_lookup_lock;
    unordered_map<labeltype, tableint> label_lookup_;

    default_random_engine level_generator_;
    default_random_engine update_probability_generator_;

    mutable atomic<long> metric_distance_computations{0};
    mutable atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  

    mutex deleted_elements_lock;  
    unordered_set<tableint> deleted_elements; 

    HierarchicalNSW(SpaceInterface<dist_t> *s): HierarchicalNSW(s, 0) {}

    HierarchicalNSW(SpaceInterface<dist_t> *s,const string &location,bool nmslib = false,size_t max_elements = 0,bool allow_replace_deleted = false): allow_replace_deleted_(allow_replace_deleted) 
    {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s,size_t max_elements,size_t M = 16,size_t ef_construction = 200,size_t random_seed = 100,bool allow_replace_deleted = false) : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),link_list_locks_(max_elements),element_levels_(max_elements),allow_replace_deleted_(allow_replace_deleted) 
    {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) 
        {
            M_ = M;
        } 
        else 
        {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." <<endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);

        if (linkLists_ == nullptr)
            throw runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");



        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    // void buildVoronoiTree(int vt_pivots,int leaf_capacity) {
    //     delete vtree_;

    //     vtree_ = new VoronoiTree(
    //         data_level0_memory_,
    //         data_size_,
    //         data_size_ / sizeof(float),
    //         cur_element_count,
    //         vt_pivots,
    //         leaf_capacity);

    //     vtree_->build();
    // }


    // Print distance between a query and one starting seed.
    // void printSeedDistance(const char* approach,const void* query_data,tableint seed_id) const 
    // {
    //     double squared_l2 = static_cast<double>(fstdistfunc_(query_data,getDataByInternalId(seed_id),dist_func_param_));
    //     std::cout<< "[SEED_DISTANCE]"<< " approach=" << approach<< " seed_internal_id=" << seed_id<< " seed_label=" << getExternalLabel(seed_id)<< " squared_l2=" << squared_l2<< " euclidean_l2=" << std::sqrt(squared_l2)<< std::endl;
    // }


    // Print distances for multiple starting seeds.
    // void printMultipleSeedDistances(const char* approach,const void* query_data,const std::vector<tableint>& seed_ids) const 
    // {
    //     if (seed_ids.empty()) 
    //     {
    //         std::cout << "[SEED_DISTANCE] approach=" << approach << " no_seeds_found" << std::endl;
    //         return;
    //     }

    //     double min_l2 = std::numeric_limits<double>::max();
    //     double max_l2 = 0.0;
    //     double sum_l2 = 0.0;

    //     std::cout << "\n[MULTI_SEED_DISTANCE]"<< " approach=" << approach << " number_of_seeds=" << seed_ids.size()<< std::endl;

    //     for (tableint seed_id : seed_ids) 
    //     {
    //         double squared_l2 = static_cast<double>(fstdistfunc_(query_data,getDataByInternalId(seed_id),dist_func_param_));
    //         double euclidean_l2 = std::sqrt(squared_l2);
    //         std::cout<< "  seed_internal_id=" << seed_id<< " seed_label=" << getExternalLabel(seed_id)<< " squared_l2=" << squared_l2<< " euclidean_l2=" << euclidean_l2<< std::endl;
    //         min_l2 = std::min(min_l2, euclidean_l2);
    //         max_l2 = std::max(max_l2, euclidean_l2);
    //         sum_l2 += euclidean_l2;
    //     }

    //     std::cout<< "  min_l2=" << min_l2<< " avg_l2=" << sum_l2 / seed_ids.size()<< " max_l2=" << max_l2<< std::endl;
    // }


    void buildMTree(int mtree_pivots,int leaf_capacity,int leafClusters) 
    {
        delete mtree_;
        mtree_ = new MTree(data_level0_memory_,size_data_per_element_,offsetData_,data_size_ / sizeof(float),cur_element_count,mtree_pivots,leaf_capacity,leafClusters);
        mtree_->build();
    }

    void buildPCTree(int numPartitions,int leaf_capacity,int leafClusters) 
    {
        delete pctree_;
        pctree_ = new ::PCTree(data_level0_memory_,size_data_per_element_,data_size_ / sizeof(float),cur_element_count,leaf_capacity,numPartitions,leafClusters);
    }

    void buildVantagePointTree(int numPartitions, int leaf_capacity, int leafClusters) 
    {
        delete vpt_;
        vpt_ = new VantagePointTree(data_level0_memory_,data_size_,data_size_ / sizeof(float),cur_element_count,leaf_capacity,numPartitions,leafClusters);
        vpt_->build();
    }

    void buildKMeansTree(int numClusters, int leafCapacity,int leafClusters)
    {
        delete kmeanstree_;
        kmeanstree_ = new KMeansTree(data_level0_memory_,size_data_per_element_,data_size_ / sizeof(float),cur_element_count,numClusters,leafCapacity,leafClusters);
        kmeanstree_->build();
    }

    void recomputeAllLeafClusters(int leafClusters)
    {
        if (mtree_ != nullptr)
            mtree_->recomputeLeafClusters(leafClusters);

        if (pctree_ != nullptr)
            pctree_->recomputeLeafClusters(leafClusters);

        if (vpt_ != nullptr)
            vpt_->recomputeLeafClusters(leafClusters);

        if (kmeanstree_ != nullptr)
            kmeanstree_->recomputeLeafClusters(leafClusters);
    }

    // Tri-Scheme precomputation
    // TRI-Scheme precomputation for Level 0 and all upper layers.
    void buildTriDistances()
    {
        tri_ready_ = false;

        if (cur_element_count == 0)
        {
            tri_edge_distances_.clear();
            upper_tri_edge_distances_.clear();
            return;
        }

        tri_edge_distances_.assign(static_cast<size_t>(cur_element_count) * maxM0_,std::numeric_limits<dist_t>::max());
        for (tableint node = 0; node < cur_element_count; ++node)
        {
            linklistsizeint* linklist = get_linklist0(node);
            size_t size = getListCount(linklist);
            tableint* neighbors = reinterpret_cast<tableint*>(linklist + 1);
            for (size_t j = 0; j < size; ++j)
            {
                tableint neighbor = neighbors[j];

                dist_t d = fstdistfunc_(
                    getDataByInternalId(node),
                    getDataByInternalId(neighbor),
                    dist_func_param_
                );

                tri_edge_distances_[
                    static_cast<size_t>(node) * maxM0_ + j
                ] = d;
            }
        }

        upper_tri_edge_distances_.clear();
        upper_tri_edge_distances_.resize(cur_element_count);

        for (tableint node = 0; node < cur_element_count; ++node)
        {
            int node_max_level = element_levels_[node];
            upper_tri_edge_distances_[node].resize(static_cast<size_t>(node_max_level) + 1);

            for (int level = 1; level <= node_max_level; ++level)
            {
                linklistsizeint* linklist = get_linklist(node, level);

                size_t size = getListCount(linklist);
                tableint* neighbors =
                    reinterpret_cast<tableint*>(linklist + 1);

                upper_tri_edge_distances_[node][level].resize(size);

                for (size_t j = 0; j < size; ++j)
                {
                    tableint neighbor = neighbors[j];
                    dist_t d = fstdistfunc_(getDataByInternalId(node),getDataByInternalId(neighbor),dist_func_param_);
                    upper_tri_edge_distances_[node][level][j] = d;
                }
            }
        }
        tri_ready_ = true;
    }



    // int getVTreeHeight() const {
    //     return (vtree_ != nullptr) ? vtree_->getHeight() : -1;
    // }

    int getMTreeHeight() const 
    {
        return (mtree_ != nullptr) ? mtree_->getHeight() : -1;
    }

    int getPCTreeHeight() const 
    {
        return (pctree_ != nullptr) ? pctree_->getHeight() : -1;
    }

    int getVPTreeHeight() const 
    {
        return (vpt_ != nullptr) ? vpt_->getHeight() : -1;
    }
    int getKMeansTreeHeight() const
    {
        return (kmeanstree_ != nullptr)? kmeanstree_->getHeight(): -1;
    }

    // Memory accounting
    // Exact, structural byte accounting for every structure built on top
    // of (or alongside) the base HNSW graph. These are computed directly
    // from live allocation sizes / container capacities, not sampled via
    // OS-level RSS, so the numbers are deterministic and repeatable.

    // MTree / PCTree / VPTree / KMeansTree: simply forward to the memory
    // walk already implemented on each tree class.
    size_t getMTreeMemoryBytes() const
    {
        return (mtree_ != nullptr) ? mtree_->memoryUsageBytes() : 0;
    }

    size_t getPCTreeMemoryBytes() const
    {
        return (pctree_ != nullptr) ? pctree_->memoryUsageBytes() : 0;
    }

    size_t getVPTreeMemoryBytes() const
    {
        return (vpt_ != nullptr) ? vpt_->memoryUsageBytes() : 0;
    }

    size_t getKMeansTreeMemoryBytes() const
    {
        return (kmeanstree_ != nullptr) ? kmeanstree_->memoryUsageBytes() : 0;
    }

    // Core HNSW graph memory: the flat level-0 buffer (vectors + labels +
    // level-0 links) plus every higher-level link list, plus the small
    // amount of bookkeeping (locks, label lookup) that HierarchicalNSW
    // itself carries. The higher-level link-list walk mirrors the exact
    // same accounting already used by indexFileSize()/saveIndex().
    size_t getHNSWBaseMemoryBytes() const
    {
        // Flat level-0 buffer: vector data + level-0 links + labels,
        // one slot per element up to max_elements_ (the true malloc size).
        size_t core_bytes = max_elements_ * size_data_per_element_;

        // Pointer table for higher-level link lists.
        core_bytes += max_elements_ * sizeof(char*);

        // Higher-level link lists themselves: only elements that were
        // actually assigned a level above 0 have an allocated buffer.
        for (size_t i = 0; i < cur_element_count; ++i)
        {
            if (element_levels_[i] > 0)
                core_bytes += size_links_per_element_ * element_levels_[i];
        }

        // element_levels_ bookkeeping vector.
        core_bytes += element_levels_.capacity() * sizeof(int);

        // Auxiliary bookkeeping: per-element/per-label locks and the
        // label -> internal-id lookup table. Kept as a clearly separate
        // addend so it can be dropped from the total if a "graph storage
        // only" number is wanted instead.
        size_t auxiliary_bytes = 0;
        auxiliary_bytes += link_list_locks_.size() * sizeof(mutex);
        auxiliary_bytes += label_op_locks_.size() * sizeof(mutex);

        // unordered_map byte accounting is allocator-dependent; approximate
        // each entry as the stored pair plus a small fixed per-node/bucket
        // overhead (next-pointer + cached hash), typical for libstdc++.
        static constexpr size_t kHashNodeOverhead = 16;
        auxiliary_bytes += label_lookup_.size() *
            (sizeof(std::pair<const labeltype, tableint>) + kHashNodeOverhead);

        return core_bytes + auxiliary_bytes;
    }

    // FINGER-specific memory: the shared projection/global data plus the
    // per-element FingerNodeData entries (each itself variable-length,
    // sized by that element's level-0 degree).
    size_t getFingerMemoryBytes() const
    {
        if (!use_finger_)
            return 0;

        size_t total = 0;

        total += finger_global_.projection.capacity() * sizeof(float);
        total += finger_global_.point_projection.capacity() * sizeof(float);
        total += finger_global_.point_norm_sq.capacity() * sizeof(float);

        total += finger_data_.capacity() * sizeof(FingerNodeData);
        for (const auto& node : finger_data_)
        {
            total += node.neighbor_coeffs.capacity() * sizeof(float);
            total += node.residual_norms.capacity() * sizeof(float);
            total += node.neighbor_center_coeffs.capacity() * sizeof(float);
        }

        return total;
    }

    // TRI-specific memory: the single flat edge-distance table.
    size_t getTriMemoryBytes() const
    {
        if (!tri_ready_)
            return 0;
        size_t total = tri_edge_distances_.capacity() * sizeof(dist_t);
        total += upper_tri_edge_distances_.capacity()* sizeof(upper_tri_edge_distances_[0]);
        for (const auto& nodeLevels : upper_tri_edge_distances_)
        {
            total += nodeLevels.capacity()* sizeof(nodeLevels[0]);
            for (const auto& levelDistances : nodeLevels)
            {
                total += levelDistances.capacity() * sizeof(dist_t);
            }
        }
        return total;
    }

    
    ~HierarchicalNSW() 
    {
        clear();
    }

    void clear() 
    {
        // delete vtree_;
        // vtree_ = nullptr;
        delete mtree_;
        mtree_ = nullptr;
        delete pctree_;
        pctree_ = nullptr;
        delete vpt_;
        vpt_ = nullptr;
        delete kmeanstree_;
        kmeanstree_ = nullptr;
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) 
        {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        tri_edge_distances_.clear();
        tri_ready_ = false;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        upper_tri_edge_distances_.clear();
    }

    struct CompareByFirst 
    {
        constexpr bool operator()(pair<dist_t, tableint> const& a,pair<dist_t, tableint> const& b) const noexcept 
        {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) 
    {
        ef_ = ef;
    }

    inline mutex& getLabelOpMutex(labeltype label) const 
    {
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const 
    {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const 
    {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const 
    {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }

    inline char *getDataByInternalId(tableint internal_id) const 
    {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    int getRandomLevel(double reverse_size) 
    {
        uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() 
    {
        return max_elements_;
    }

    size_t getCurrentElementCount() 
    {
        return cur_element_count;
    }

    size_t getDeletedCount() 
    {
        return num_deleted_;
    }

    priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> searchBaseLayer(tableint ep_id, const void *data_point, int layer) 
    {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) 
        {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } 
        else 
        {
            lowerBound = numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) 
        {
            pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) 
            {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            unique_lock <mutex> lock(link_list_locks_[curNodeNum]);

            int *data;
            
            if (layer == 0) 
            {
                data = (int*)get_linklist0(curNodeNum);
            } 
            else 
            {
                data = (int*)get_linklist(curNodeNum, layer);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);

            for (size_t j = 0; j < size; j++) 
            {
                tableint candidate_id = *(datal + j);
                if (visited_array[candidate_id] == visited_array_tag) 
                    continue;

                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) 
                {
                    candidateSet.emplace(-dist1, candidate_id);

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }



    // Level 0 single entry point
    template <bool bare_bone_search = true, bool collect_metrics = false,bool collect_cycles = false>priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> searchBaseLayerST(tableint ep_id,const void *data_point,size_t ef,uint64_t &distanceCounter,uint64_t &distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,BaseSearchStopCondition<dist_t>* stop_condition = nullptr,profilingStats* profiler = nullptr) const 
    {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) 
        {
            char* ep_data = getDataByInternalId(ep_id);

            uint64_t distanceCyclesStart = 0;


            // Distance Computation
            if constexpr (collect_cycles)
                distanceCyclesStart = profiler->readCpuCycles();
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            if constexpr (collect_cycles)
                distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
            distanceCounter++;



            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) 
            {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } 
        else 
        {
            lowerBound = numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) 
        {
            pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) 
            {
                flag_stop_search = candidate_dist > lowerBound;
            } 
            else 
            {
                if (stop_condition) 
                {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } 
                else 
                {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            
            if (flag_stop_search) 
            {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            if (collect_metrics) 
            {
                metric_hops++;
                metric_distance_computations+=size;
            }

            for (size_t j = 1; j <= size; j++) 
            {
                int candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) 
                {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));



                    // Distance Computation
                    uint64_t distanceCyclesStart = 0;
                    if constexpr (collect_cycles)
                    {
                        distanceCyclesStart = profiler->readCpuCycles();
                    }
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if constexpr (collect_cycles)
                    {
                        distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                    }
                    distanceCounter++;



                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) 
                    {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } 
                    else 
                    {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) 
                    {
                        candidate_set.emplace(-dist, candidate_id);

                        if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) 
                        {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) 
                            {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        
                        if (!bare_bone_search && stop_condition) 
                        {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } 
                        else 
                        {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) 
                        {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) 
                            {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } 
                            else 
                            {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // level 0 multiple entry point
    template <bool bare_bone_search = true> priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst>searchBaseLayerSTMulti(const std::vector<tableint>& ep_ids,const void *data_point,size_t ef,uint64_t &distanceCounter,uint64_t &distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,BaseSearchStopCondition<dist_t>* stop_condition = nullptr,profilingStats* profiler = nullptr) const
    {   
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> candidate_set;

       
        for (tableint ep_id : ep_ids)
        {
            if (ep_id >= cur_element_count)
                continue;
            if (visited_array[ep_id] == visited_array_tag)
                continue;
            visited_array[ep_id] = visited_array_tag;
            char* ep_data = getDataByInternalId(ep_id);

            const uint64_t distanceCyclesStart = profiler->readCpuCycles();
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
            distanceCounter++;
            candidate_set.emplace(-dist, ep_id);
            if (bare_bone_search || (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))))
            {
                top_candidates.emplace(dist, ep_id);
                if (!bare_bone_search && stop_condition)
                {
                    stop_condition->add_point_to_result(getExternalLabel(ep_id),ep_data,dist
                    );
                }
            }
        }
        while (top_candidates.size() > ef)
        {
            top_candidates.pop();
        }
        dist_t lowerBound =top_candidates.empty()? numeric_limits<dist_t>::max(): top_candidates.top().first;
        while (!candidate_set.empty())
        {
            pair<dist_t, tableint> current_node_pair =
                candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            if (bare_bone_search)
            {
                flag_stop_search = candidate_dist > lowerBound;
            }
            else
            {
                if (stop_condition)
                {
                    flag_stop_search =stop_condition->should_stop_search(candidate_dist,lowerBound);
                }
                else
                {
                    flag_stop_search =candidate_dist > lowerBound &&top_candidates.size() == ef;
                }
            }

            if (flag_stop_search)
            {
                break;
            }
            candidate_set.pop();
            tableint current_node_id =current_node_pair.second;

            int *data =(int *)get_linklist0(current_node_id);

            size_t size =getListCount((linklistsizeint*)data);

            for (size_t j = 1; j <= size; j++)
            {
                int candidate_id = *(data + j);
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;
                visited_array[candidate_id] = visited_array_tag;

                char *currObj1 = getDataByInternalId(candidate_id);
                const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                distanceCounter++;

                bool flag_consider_candidate;
                if (!bare_bone_search && stop_condition)
                {
                    flag_consider_candidate =stop_condition->should_consider_candidate(dist,lowerBound);
                }
                else
                {
                    flag_consider_candidate =top_candidates.size() < ef ||lowerBound > dist;
                }

                if (flag_consider_candidate)
                {
                    candidate_set.emplace(-dist,candidate_id);
                    if (bare_bone_search || (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) ||(*isIdAllowed)(getExternalLabel(candidate_id)))))
                    {
                        top_candidates.emplace(dist,candidate_id);
                        if (!bare_bone_search && stop_condition)
                        {
                            stop_condition->add_point_to_result(getExternalLabel(candidate_id),currObj1,dist);
                        }
                    }
                    bool flag_remove_extra = false;
                    if (!bare_bone_search && stop_condition)
                    {
                        flag_remove_extra =stop_condition->should_remove_extra();
                    }
                    else
                    {
                        flag_remove_extra =top_candidates.size() > ef;
                    }
                    while (flag_remove_extra)
                    {
                        tableint id =top_candidates.top().second;
                        top_candidates.pop();
                        if (!bare_bone_search && stop_condition)
                        {
                            stop_condition->remove_point_from_result(getExternalLabel(id),getDataByInternalId(id),dist);
                            flag_remove_extra =stop_condition->should_remove_extra();
                        }
                        else
                        {
                            flag_remove_extra =top_candidates.size() > ef;
                        }
                    }
                    if (!top_candidates.empty())
                        lowerBound =top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    // FINGER-ACCELERATED BASE LAYER SEARCH METHOD
    priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst>searchBaseLayerFinger(tableint ep_id,const void* query_data,size_t ef,uint64_t& distanceCounter,uint64_t& distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const
    {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        const int dim = static_cast<int>(data_size_ / sizeof(float));
        const float* query = reinterpret_cast<const float*>(query_data);
        const float query_norm_sq = finger_norm_sq(query, dim);

        std::vector<float> query_projection(finger_rank_);
        finger_project(query,finger_global_.projection.data(),dim,finger_rank_,query_projection.data());

        dist_t lowerBound;
        
        if (!isMarkedDeleted(ep_id) && (!isIdAllowed || (*isIdAllowed)(getExternalLabel(ep_id)))) 
        {
            uint64_t distanceCyclesStart = profiler->readCpuCycles();
            dist_t dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
            distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;

            distanceCounter++;
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } 
        else 
        {
            lowerBound = numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;
        size_t updates = 0;

        while (!candidate_set.empty()) 
        {
            pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            if (candidate_dist > lowerBound && top_candidates.size() == ef)
                break;

            candidate_set.pop();
            tableint curr_node = current_node_pair.second;
            ++updates;

            int *data = (int *) get_linklist0(curr_node);
            size_t size = getListCount((linklistsizeint*)data);
            const float* center = reinterpret_cast<const float*>(getDataByInternalId(curr_node));
            float center_norm_sq = finger_global_.point_norm_sq[curr_node];
            const float* center_projection = &finger_global_.point_projection[static_cast<size_t>(curr_node) * finger_rank_];

            float alpha = 0.0f;
            if (center_norm_sq > 1e-20f) 
            {
                alpha = (query_norm_sq + center_norm_sq - static_cast<float>(candidate_dist)) / (2.0f * center_norm_sq);
            }

            const FingerNodeData& node = finger_data_[curr_node];
            bool use_approx = updates > finger_warmup_updates_ && node.neighbor_coeffs.size() >= size * static_cast<size_t>(finger_rank_);

            for (size_t j = 0; j < size; ++j) 
            {
                tableint cand_id = *(data + j + 1);
                if (visited_array[cand_id] == visited_array_tag) 
                    continue;
                visited_array[cand_id] = visited_array_tag;

                dist_t dist;
                if (use_approx) 
                {
                    float est = finger_approx_distance(query_norm_sq,center_norm_sq,static_cast<float>(candidate_dist),query_projection.data(),center_projection,&node.neighbor_coeffs[j * static_cast<size_t>(finger_rank_)],node.residual_norms[j],node.neighbor_center_coeffs[j],finger_global_);

                    if (top_candidates.size() >= ef && static_cast<float>(lowerBound) < est)
                        continue;
                }

                uint64_t distanceCyclesStart = 0;
                distanceCyclesStart = profiler->readCpuCycles();
                dist = fstdistfunc_(query_data, getDataByInternalId(cand_id), dist_func_param_);
                distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                distanceCounter++;
                if (top_candidates.size() < ef || lowerBound > dist) 
                {
                    candidate_set.emplace(-dist, cand_id);

                    if (!isMarkedDeleted(cand_id) && (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))))
                        top_candidates.emplace(dist, cand_id);

                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }



    // TRIANGLE-ACCELERATED BASE LAYER SEARCH
    priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst>searchBaseLayerTri(tableint ep_id,const void* query_data,size_t ef,uint64_t& distanceCounter,uint64_t& distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const
    {
        if (!tri_ready_)
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst>top_candidates;
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst>candidate_set;


        // Distance from query to entry point

        uint64_t distanceCyclesStart = profiler->readCpuCycles();
        dist_t ep_dist = fstdistfunc_(query_data,getDataByInternalId(ep_id),dist_func_param_);
        distanceCycles +=profiler->readCpuCycles() - distanceCyclesStart;


        distanceCounter++;


        dist_t lowerBound = ep_dist;
        top_candidates.emplace(ep_dist, ep_id);
        candidate_set.emplace(-ep_dist, ep_id);
        visited_array[ep_id] = visited_array_tag;
        while (!candidate_set.empty())
        {
            pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            // Standard HNSW termination condition
            if (candidate_dist > lowerBound && top_candidates.size() == ef)
            {
                break;
            }
            candidate_set.pop();
            tableint curr_node = current_node_pair.second;
            int *data = (int *)get_linklist0(curr_node);
            size_t size = getListCount((linklistsizeint*)data);

            // d_qc = squared distance(q, curr_node)
            dist_t d_qc = candidate_dist;

            // Convert to Euclidean distance once for this curr_node.
            // This is required because fstdistfunc_ returns squared L2.
            dist_t sqrt_qc = std::sqrt(d_qc);
            for (size_t j = 0; j < size; ++j)
            {
                tableint cand_id = *(data + j + 1);
                if (visited_array[cand_id] == visited_array_tag)
                {
                    continue;
                }
                visited_array[cand_id] = visited_array_tag;
                // d_cc = squared distance(curr_node, cand_id)
                dist_t d_cc = tri_edge_distances_[static_cast<size_t>(curr_node) * maxM0_ + j];

                // Convert edge distance to Euclidean distance.
                dist_t sqrt_cc = std::sqrt(d_cc);

                // Valid triangle-inequality lower bound for squared L2:
                // ||q - cand||^2 >= (||q - curr|| - ||curr - cand||)^2

                dist_t diff = sqrt_qc - sqrt_cc;
                dist_t tri_lb = diff * diff;

                // TRI pruning
                if (top_candidates.size() >= ef && tri_lb >= lowerBound)
                {
                    continue;
                }

                // Exact squared-L2 distance

                const uint64_t neighborDistanceCyclesStart = profiler->readCpuCycles();
                dist_t dist = fstdistfunc_(query_data,getDataByInternalId(cand_id),dist_func_param_);
                distanceCycles += profiler->readCpuCycles() - neighborDistanceCyclesStart;

                distanceCounter++;

                if (top_candidates.size() < ef ||lowerBound > dist)
                {
                    candidate_set.emplace(-dist, cand_id);
                    if (!isMarkedDeleted(cand_id) && (!isIdAllowed || (*isIdAllowed)( getExternalLabel(cand_id))))
                    {
                        top_candidates.emplace(dist, cand_id);
                    }
                    if (top_candidates.size() > ef)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }



    void getNeighborsByHeuristic2(priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> &top_candidates,const size_t M) 
    {
        if (top_candidates.size() < M) 
        {
            return;
        }

        priority_queue<pair<dist_t, tableint>> queue_closest;
        vector<pair<dist_t, tableint>> return_list;
        
        while (top_candidates.size() > 0) 
        {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) 
        {
            if (return_list.size() >= M)
                break;
            pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;
            for (pair<dist_t, tableint> second_pair : return_list) 
            {
                dist_t curdist =fstdistfunc_(getDataByInternalId(second_pair.second),getDataByInternalId(curent_pair.second),dist_func_param_);
                if (curdist < dist_to_query) 
                {
                    good = false;
                    break;
                }
            }
            if (good) 
            {
                return_list.push_back(curent_pair);
            }
        }

        for (pair<dist_t, tableint> curent_pair : return_list) 
        {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const 
    {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const 
    {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const 
    {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const 
    {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint mutuallyConnectNewElement(const void *data_point,tableint cur_c,priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> &top_candidates,int level,bool isUpdate) 
    {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) 
        {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();
        {
            unique_lock <mutex> lock(link_list_locks_[cur_c], defer_lock);
            if (isUpdate) 
            {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) 
            {
                throw runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) 
            {
                if (data[idx] && !isUpdate)
                    throw runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) 
        {
            unique_lock <mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) 
            {
                for (size_t j = 0; j < sz_link_list_other; j++) 
                {
                    if (data[j] == cur_c) 
                    {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            if (!is_cur_c_present) 
            {
                if (sz_link_list_other < Mcurmax) 
                {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } 
                else 
                {
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),dist_func_param_);
                    priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);
                    
                    for (size_t j = 0; j < sz_link_list_other; j++) 
                    {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) 
                    {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) 
    {
        if (new_max_elements < cur_element_count)
            throw runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        vector<mutex>(new_max_elements).swap(link_list_locks_);

        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const 
    {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) 
        {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) 
    {
        ofstream output(location, ios::binary);

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) 
        {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) 
    {
        ifstream input(location, ios::binary);

        if (!input.is_open())
            throw runtime_error("Cannot open file");

        clear();
        input.seekg(0, input.end);
        streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;

        if (max_elements < cur_element_count)
            max_elements = max_elements_;


        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) 
        {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) 
            {
                throw runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) 
            {
                input.seekg(linkListSize, input.cur);
            }
        }

        if (input.tellg() != total_filesize)
            throw runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        vector<mutex>(max_elements).swap(link_list_locks_);
        vector<mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);

        if (linkLists_ == nullptr)
            throw runtime_error("Not enough memory: loadIndex failed to allocate linklists");


        element_levels_ = vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) 
        {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } 
            else 
            {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) 
        {
            if (isMarkedDeleted(i)) 
            {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();
        return;
    }

    template<typename data_t> vector<data_t> getDataByLabel(labeltype label) const 
    {
        unique_lock <mutex> lock_label(getLabelOpMutex(label));    
        unique_lock <mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) 
        {
            throw runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();
        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) 
        {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    void markDelete(labeltype label) 
    {
        unique_lock <mutex> lock_label(getLabelOpMutex(label));
        unique_lock <mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) 
        {
            throw runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();
        markDeletedInternal(internalId);
    }

    void markDeletedInternal(tableint internalId) 
    {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) 
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) 
            {
                unique_lock <mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } 
        else 
        {
            throw runtime_error("The requested to delete element is already deleted");
        }
    }

    void unmarkDelete(labeltype label) 
    {
        unique_lock <mutex> lock_label(getLabelOpMutex(label));
        unique_lock <mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) 
        {
            throw runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();
        unmarkDeletedInternal(internalId);
    }

    void unmarkDeletedInternal(tableint internalId) 
    {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) 
        {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) 
            {
                unique_lock <mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } 
        else 
        {
            throw runtime_error("The requested to undelete element is not deleted");
        }
    }

    bool isMarkedDeleted(tableint internalId) const 
    {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint * ptr) const 
    {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint * ptr, unsigned short int size) const 
    {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) 
    {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) 
        {
            throw runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        unique_lock <mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) 
        {
            addPoint(data_point, label, -1);
            return;
        }

        tableint internal_id_replaced;
        unique_lock <mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) 
        {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        if (!is_vacant_place) 
        {
            addPoint(data_point, label, -1);
        } 
        else 
        {
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);
            unique_lock <mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) 
    {
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);
        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;

        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) 
        {
            unordered_set<tableint> sCand;
            unordered_set<tableint> sNeigh;
            vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) 
            {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                
                for (auto&& elTwoHop : listTwoHop) 
                {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) 
            {
                priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;
                size_t elementsToKeep = min(ef_construction_, size);
                for (auto&& cand : sCand) 
                {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) 
                    {
                        candidates.emplace(distance, cand);
                    } 
                    else 
                    {
                        if (distance < candidates.top().first) 
                        {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);
                {
                    unique_lock <mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) 
                    {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(const void *dataPoint,tableint entryPointInternalId,tableint dataPointInternalId,int dataPointLevel,int maxLevel) 
    {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) 
        {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) 
            {
                bool changed = true;
                while (changed) 
                {
                    changed = false;
                    unsigned int *data;
                    unique_lock <mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);

                    for (int i = 0; i < size; i++) 
                    {
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) 
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) 
        {
            priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(currObj, dataPoint, level);

            priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) 
            {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());
                topCandidates.pop();
            }

            if (filteredTopCandidates.size() > 0) 
            {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) 
                {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }
                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }

    vector<tableint> getConnectionsWithLock(tableint internalId, int level) 
    {
        unique_lock <mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) 
    {
        tableint cur_c = 0;
        {
            unique_lock <mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) 
            {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) 
                {
                    if (isMarkedDeleted(existingInternalId)) 
                    {
                        throw runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();
                if (isMarkedDeleted(existingInternalId)) 
                {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);
                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) 
            {
                throw runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        unique_lock <mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        unique_lock <mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel > 0) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel);
            if (linkLists_[cur_c] == nullptr)
                throw runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        unique_lock <mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw runtime_error("Level error");

                priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }

        if (mtree_ != nullptr) {
            mtree_->insert(static_cast<int>(cur_c));
        }

        return cur_c;
    }


    // FINGER search with single entry points
    std::priority_queue<std::pair<dist_t, labeltype>>searchFromEntryPointFinger(tableint entry_point,const void* query_data,size_t k,uint64_t& distanceCounter,uint64_t& distanceCycles,BaseFilterFunctor* isIdAllowed,profilingStats* profiler) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) 
            return result;

        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        if (!use_finger_ || !finger_global_.ready) 
        {
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            if (bare_bone_search) 
            {
                top_candidates = searchBaseLayerST<true, false, true>(entry_point, query_data, max(ef_, k),distanceCounter, distanceCycles,isIdAllowed, nullptr, profiler);
            } 
            else 
            {
                top_candidates = searchBaseLayerST<false>(entry_point, query_data, max(ef_, k),distanceCounter, distanceCycles,isIdAllowed, nullptr, profiler);
            }
        } 
        else 
        {
            top_candidates = searchBaseLayerFinger(entry_point, query_data, max(ef_, k),distanceCounter, distanceCycles,isIdAllowed, profiler);
        }

        while (top_candidates.size() > k)
            top_candidates.pop();

        while (!top_candidates.empty()) 
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push(pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }    
        
        return result;
    }    


    // TRI search with single entry points
    std::priority_queue<std::pair<dist_t, labeltype>>searchFromEntryPointTri(tableint entry_point,const void* query_data,size_t k,uint64_t &distanceCounter,uint64_t &distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const 
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;

        if (!tri_ready_) 
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerTri(entry_point,query_data,max(ef_, k),distanceCounter,distanceCycles,isIdAllowed,profiler);

        while (top_candidates.size() > k)
            top_candidates.pop();

        while (!top_candidates.empty()) 
        {
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push(pair<dist_t, labeltype>(rez.first,getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    // FINGER search with multiple entry points
    std::priority_queue<std::pair<dist_t, labeltype>>searchFromEntryPointFingerMulti(const std::vector<tableint>& entry_points,const void* query_data,size_t k,uint64_t& distanceCounter,uint64_t& distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0 || entry_points.empty())
            return result;

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;    

        // If FINGER is not available, use normal multi-entry search.
        if (!use_finger_ || !finger_global_.ready)
        {
            bool bare_bone_search = !num_deleted_ && !isIdAllowed;
            if (bare_bone_search)
            {
                top_candidates = searchBaseLayerSTMulti<true>(entry_points, query_data, max(ef_, k),distanceCounter, distanceCycles,isIdAllowed, nullptr, profiler);
            }    
            else
            {
                top_candidates = searchBaseLayerSTMulti<false>(entry_points, query_data, max(ef_, k),distanceCounter, distanceCycles,isIdAllowed, nullptr, profiler);
            }    
        }    
        else
        {
            //  Multi-entry FINGER Level-0 search.
            // All entry points are inserted into the same candidate
            // queues and searched together.
            VisitedList* vl = visited_list_pool_->getFreeVisitedList();
            vl_type* visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> candidate_set;

            const int dim =static_cast<int>(data_size_ / sizeof(float));

            const float* query = reinterpret_cast<const float*>(query_data);

            const float query_norm_sq = finger_norm_sq(query, dim);

            std::vector<float> query_projection(finger_rank_);

            finger_project(query,finger_global_.projection.data(),dim,finger_rank_,query_projection.data());

            dist_t lowerBound =numeric_limits<dist_t>::max();
            for (tableint ep_id : entry_points)
            {
                if (ep_id >= cur_element_count)
                    continue;

                if (visited_array[ep_id] == visited_array_tag)    
                    continue;

                visited_array[ep_id] = visited_array_tag;    

                const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                dist_t dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
                distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                distanceCounter++;

                candidate_set.emplace(-dist, ep_id);
                if (!isMarkedDeleted(ep_id) && (!isIdAllowed ||(*isIdAllowed)(getExternalLabel(ep_id))))
                {
                    top_candidates.emplace(dist, ep_id);
                }    
            }    

            while (top_candidates.size() > max(ef_, k))
                top_candidates.pop();

            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;

            size_t updates = 0;    

            while (!candidate_set.empty())
            {
                pair<dist_t, tableint> current_node_pair = candidate_set.top();

                dist_t candidate_dist = -current_node_pair.first;

                if (candidate_dist > lowerBound && top_candidates.size() == max(ef_, k))
                {
                    break;
                }    

                candidate_set.pop();
                tableint curr_node  = current_node_pair.second;

                ++updates;

                int* data = (int*)get_linklist0(curr_node);

                size_t size = getListCount((linklistsizeint*)data);

                const float* center = reinterpret_cast<const float*>(getDataByInternalId(curr_node));

                float center_norm_sq = finger_global_.point_norm_sq[curr_node];

                const float* center_projection = &finger_global_.point_projection[static_cast<size_t>(curr_node) * finger_rank_];

                float alpha = 0.0f;

                if (center_norm_sq > 1e-20f)
                {
                    alpha =(query_norm_sq +center_norm_sq - static_cast<float>(candidate_dist))/ (2.0f * center_norm_sq);
                }    
                const FingerNodeData& node = finger_data_[curr_node];
                bool use_approx =updates > finger_warmup_updates_ && node.neighbor_coeffs.size() >= size * static_cast<size_t>(finger_rank_);

                for (size_t j = 0; j < size; ++j)
                {
                    tableint cand_id = *(data + j + 1);

                    if (visited_array[cand_id] == visited_array_tag)
                    {
                        continue;
                    }    

                    visited_array[cand_id] = visited_array_tag;

                    if (use_approx)
                    {
                        float est =finger_approx_distance(query_norm_sq,center_norm_sq,static_cast<float>(candidate_dist),query_projection.data(),center_projection,&node.neighbor_coeffs[j * static_cast<size_t>(finger_rank_)],node.residual_norms[j],node.neighbor_center_coeffs[j],finger_global_);

                        if (top_candidates.size() >= max(ef_, k) &&
                            static_cast<float>(lowerBound) < est)
                        {
                            continue;
                        }    
                    }    

                    const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                    dist_t dist = fstdistfunc_(query_data, getDataByInternalId(cand_id), dist_func_param_);distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                    distanceCounter++;

                    if (top_candidates.size() < max(ef_, k) || lowerBound > dist)
                    {
                        candidate_set.emplace(-dist, cand_id);
                        if (!isMarkedDeleted(cand_id) &&(!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))))
                        {
                            top_candidates.emplace(dist,cand_id);
                        }    

                        if (top_candidates.size() > max(ef_, k))
                            top_candidates.pop();

                        if (!top_candidates.empty())    
                            lowerBound = top_candidates.top().first;
                    }        
                }    
            }    

            visited_list_pool_->releaseVisitedList(vl);
        }    

        while (top_candidates.size() > k)
            top_candidates.pop();

        while (!top_candidates.empty())    
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push(pair<dist_t, labeltype>(rez.first,getExternalLabel(rez.second)));
            top_candidates.pop();
        }    
        return result;
    }    


    // TRI search with multiple entry points
    std::priority_queue<std::pair<dist_t, labeltype>> searchFromEntryPointTriMulti(const std::vector<tableint>& entry_points,const void* query_data,size_t k,uint64_t &distanceCounter,uint64_t &distanceCycles,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0 || entry_points.empty())
            return result;

        if (!tri_ready_)
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> candidate_set;
        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;
        const size_t ef_search = max(ef_, k);
        for (tableint ep_id : entry_points)
        {
            if (ep_id >= cur_element_count)
                continue;

            if (visited_array[ep_id] == visited_array_tag)
                continue;

            visited_array[ep_id] = visited_array_tag;
            const uint64_t distanceCyclesStart = profiler->readCpuCycles();
            dist_t dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
            distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
            distanceCounter++;

            candidate_set.emplace(-dist, ep_id);
            if (!isMarkedDeleted(ep_id) &&(!isIdAllowed || (*isIdAllowed)(getExternalLabel(ep_id))))
            {
                top_candidates.emplace(dist, ep_id);
            }
        }
        while (top_candidates.size() > ef_search)
            top_candidates.pop();
        dist_t lowerBound =top_candidates.empty()? numeric_limits<dist_t>::max(): top_candidates.top().first;
        while (!candidate_set.empty())
        {
            pair<dist_t, tableint> current_node_pair =candidate_set.top();
            dist_t candidate_dist =-current_node_pair.first;
            if (candidate_dist > lowerBound && top_candidates.size() == ef_search)
            {
                break;
            }
            candidate_set.pop();
            tableint curr_node = current_node_pair.second;
            int* data = (int*)get_linklist0(curr_node);
            size_t size = getListCount((linklistsizeint*)data);
            dist_t d_qc = candidate_dist;
            dist_t sqrt_qc = std::sqrt(d_qc);

            for (size_t j = 0; j < size; ++j)
            {
                tableint cand_id = *(data + j + 1);
                if (visited_array[cand_id] == visited_array_tag)
                {
                    continue;
                }
                visited_array[cand_id] = visited_array_tag;
                dist_t d_cc =tri_edge_distances_[static_cast<size_t>(curr_node) * maxM0_ + j];
                dist_t sqrt_cc = std::sqrt(d_cc);
                dist_t diff = sqrt_qc - sqrt_cc;
                dist_t tri_lb = diff * diff;

                if (top_candidates.size() >= ef_search && tri_lb >= lowerBound)
                {
                    continue;
                }

                const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                dist_t dist = fstdistfunc_(query_data, getDataByInternalId(cand_id), dist_func_param_);
                distanceCycles += profiler->readCpuCycles() - distanceCyclesStart;
                distanceCounter++;

                if (top_candidates.size() < ef_search || lowerBound > dist)
                {
                    candidate_set.emplace(-dist,cand_id);
                    if (!isMarkedDeleted(cand_id) && (!isIdAllowed || (*isIdAllowed)(getExternalLabel(cand_id))))
                    {
                        top_candidates.emplace(dist,cand_id);
                    }
                    if (top_candidates.size() > ef_search)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push(pair<dist_t, labeltype>(rez.first,getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }



    // std::priority_queue<std::pair<dist_t, labeltype>>
    // searchKnnVTree(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr) const 
    // {
    //     priority_queue<pair<dist_t, labeltype>> result;
    //     if (cur_element_count == 0) return result;

    //     tableint currObj = enterpoint_node_;
    //     dist_t   curdist = fstdistfunc_(query_data,
    //                                      getDataByInternalId(enterpoint_node_),
    //                                      dist_func_param_);

    //     if (vtree_ != nullptr) {
    //         const float* query_f = reinterpret_cast<const float*>(query_data);
    //         const int seed = vtree_->searchEntryPoint(query_f);

    //         if (seed >= 0 &&
    //             static_cast<size_t>(seed) < cur_element_count)
    //         {
    //             dist_t d = fstdistfunc_(
    //                 query_data,
    //                 getDataByInternalId(static_cast<tableint>(seed)),
    //                 dist_func_param_);

    //             if (d < curdist)
    //             {
    //                 curdist = d;
    //                 currObj = static_cast<tableint>(seed);
    //             }
    //         }
    //     }

    //     priority_queue<pair<dist_t, tableint>,
    //                         vector<pair<dist_t, tableint>>,
    //                         CompareByFirst> top_candidates;

    //     bool bare_bone_search = !num_deleted_ && !isIdAllowed;
    //     if (bare_bone_search) {
    //         top_candidates = searchBaseLayerST<true>(
    //                 currObj, query_data, max(ef_, k), isIdAllowed);
    //     } else {
    //         top_candidates = searchBaseLayerST<false>(
    //                 currObj, query_data, max(ef_, k), isIdAllowed);
    //     }

    //     while (top_candidates.size() > k) {
    //         top_candidates.pop();
    //     }
    //     while (!top_candidates.empty()) {
    //         pair<dist_t, tableint> rez = top_candidates.top();
    //         result.push({rez.first, getExternalLabel(rez.second)});
    //         top_candidates.pop();
    //     }
    //     return result;
    // }


//     std::priority_queue<std::pair<dist_t, labeltype>>
//     searchKnnVTreeFinger(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr) const 
//     {
//         priority_queue<pair<dist_t, labeltype>> empty;
//         if (cur_element_count == 0) return empty;

//         tableint currObj = enterpoint_node_;
//         dist_t curdist = fstdistfunc_(query_data,
//                                       getDataByInternalId(enterpoint_node_),
//                                       dist_func_param_);

//         if (vtree_ != nullptr) {
//             const float* query_f = reinterpret_cast<const float*>(query_data);
//            const int seed = vtree_->searchEntryPoint(query_f);

//             if (seed >= 0 &&
//                 static_cast<size_t>(seed) < cur_element_count)
//             {
//                 dist_t d = fstdistfunc_(
//                     query_data,
//                     getDataByInternalId(static_cast<tableint>(seed)),
//                     dist_func_param_);

//                 if (d < curdist)
//                 {
//                     curdist = d;
//                     currObj = static_cast<tableint>(seed);
//                 }
// }
//         }
//         return searchFromEntryPointFinger(currObj, query_data, k, isIdAllowed);
//     }



    // MTree approaches
    // MTree - Normal HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnMTree(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) 
            return result;

        tableint currObj = enterpoint_node_;
        

        const uint64_t hnswMTreeUpperLayerCyclesStart = profiler->readCpuCycles();
        if (mtree_ != nullptr) 
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            auto tree_start = std::chrono::high_resolution_clock::now();
            const int seed = mtree_->searchEntryPoint(query_f,profiler->mtreeDistanceComputations,profiler->hnswMTreeDistanceCycles,beamWidth,profiler);
            auto tree_end = std::chrono::high_resolution_clock::now();
            profiler->mtreeTimeNs += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
            
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count) 
            {    
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->hnswMTreeUpperLayerCycles += profiler->readCpuCycles() - hnswMTreeUpperLayerCyclesStart;


        // Have to comment out next line
        // printSeedDistance("MTree",query_data,currObj);

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t hnswMTreeLevel0CyclesStart = profiler->readCpuCycles();
        if (bare_bone_search) 
        {
            top_candidates = searchBaseLayerST<true, false, true>(currObj,query_data,max(ef_, k),profiler->hnswDistanceComputations_MTree,profiler->hnswMTreeDistanceCycles,isIdAllowed,nullptr,profiler);
        } 
        else 
        {
            top_candidates = searchBaseLayerST<false, false, true>(currObj,query_data,max(ef_, k),profiler->hnswDistanceComputations_MTree,profiler->hnswMTreeDistanceCycles,isIdAllowed,nullptr,profiler);
        }
        while (top_candidates.size() > k) 
        {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) 
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push({rez.first, getExternalLabel(rez.second)});
            top_candidates.pop();
        }
        profiler->hnswMTreeLevel0Cycles +=profiler->readCpuCycles() - hnswMTreeLevel0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0_MTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
        return result;
    }

    // MTree - FINGER HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnMTreeFinger(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0) 
            return empty;

        tableint currObj = enterpoint_node_;
    

        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (mtree_ != nullptr) 
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter=0;
            const int seed = mtree_->searchEntryPoint(query_f,ignoredCounter,profiler->fingerMTreeDistanceCycles,1,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count) 
            {
               currObj = static_cast<tableint>(seed);
            }
        }
        profiler->fingerMTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFinger(currObj, query_data, k,profiler->hnswFingerDistanceComputations_MTree,profiler->fingerMTreeDistanceCycles,isIdAllowed, profiler);

        profiler->fingerMTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0_MTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
        
        return result;
    }

    // MTree - Tri HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnMTreeTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (!tri_ready_) 
        {
            throw runtime_error(
                "TRI search requested before buildTriDistances().");
        }

        tableint currObj = enterpoint_node_;
        
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (mtree_ != nullptr) 
        {
            const float* query_f =reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter=0;
            const int seed = mtree_->searchEntryPoint(query_f,profiler->mtreeDistanceComputations,profiler->triMTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 &&static_cast<size_t>(seed) < cur_element_count) 
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->triMTreeUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointTri(currObj, query_data, k,profiler->hnswTriDistanceComputations_MTree,profiler->triMTreeDistanceCycles,isIdAllowed, profiler);

        profiler->triMTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0_MTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
        
        return result;
    }

    // MTree -> HNSW Normal with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnMTreeMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;
        if (mtree_ == nullptr)
            return result;
        const float* query_f =reinterpret_cast<const float*>(query_data);
    
        // Get multiple entry points from MTree
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = mtree_->searchEntryPointMulti(query_f,profiler->mtreeDistanceComputationsMulti,profiler->hnswMTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->hnswMTreeMultiUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;
        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        // Fallback if no valid MTree entry points exist
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }


        // Have to comment out next line
        // printMultipleSeedDistances("MTree-Multi",query_data,entry_points);

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        
        
        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();
        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerSTMulti<true>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_MTreeMulti,profiler->hnswMTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        else
        {
            top_candidates = searchBaseLayerSTMulti<false>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_MTreeMulti,profiler->hnswMTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push({rez.first,getExternalLabel(rez.second)});
            top_candidates.pop();
        }
        

        profiler->hnswMTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0MTreeMulti += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
        
        return result;
    }


    // MTree -> HNSW Finger with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnMTreeFingerMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (mtree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        uint64_t ignoredCounter = 0;
        std::vector<int> seeds = mtree_->searchEntryPointMulti(query_f,ignoredCounter,profiler->fingerMTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->fingerMTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;


        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 &&
                static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW Finger search using ALL selected entry points


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointFingerMulti(entry_points, query_data, k,profiler->hnswFingerDistanceComputations_MTreeMulti,profiler->fingerMTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->fingerMTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0MTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }

    // MTree -> HNSW Tri with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnMTreeTriMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (mtree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        uint64_t ignoredCounter = 0;
        std::vector<int> seeds = mtree_->searchEntryPointMulti(query_f,ignoredCounter,profiler->triMTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->triMTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);

                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW TRI search using ALL selected entry points
        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointTriMulti(entry_points, query_data, k,profiler->hnswTriDistanceComputations_MTreeMulti,profiler->triMTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->triMTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0MTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }


    // VPTree approaches
    // VPTree - Normal HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnVPTree(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {

        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) 
            return result;

        tableint currObj = enterpoint_node_;


        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (vpt_ != nullptr) 
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);

            auto tree_start = std::chrono::high_resolution_clock::now();
            const int seed = vpt_->searchEntryPoint(query_f,profiler->vptreeDistanceComputations,profiler->hnswVPTreeDistanceCycles,beamWidth,profiler);
            auto tree_end =std::chrono::high_resolution_clock::now();
            if (profiler != nullptr)
            {
                profiler->vptreeTimeNs += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
            }


            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->hnswVPTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        // Have to comment out next line
        // printSeedDistance("VPTree",query_data,currObj);

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerST<true, false, true>(currObj,query_data,max(ef_, k),profiler->hnswDistanceComputations_VPTree,profiler->hnswVPTreeDistanceCycles,isIdAllowed,nullptr,profiler);
        }
        else
        {
            top_candidates = searchBaseLayerST<false, false, true>(currObj,query_data,max(ef_, k),profiler->hnswDistanceComputations_VPTree,profiler->hnswVPTreeDistanceCycles,isIdAllowed,nullptr,profiler);
        }
        while (top_candidates.size() > k) 
        {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) 
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push({rez.first, getExternalLabel(rez.second)});
            top_candidates.pop();
        }
        profiler->hnswVPTreeLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0_VPTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // VPTree - FINGER HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnVPTreeFinger(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0) 
            return empty;

        tableint currObj = enterpoint_node_;
       
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        if (vpt_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;

            const int seed = vpt_->searchEntryPoint(query_f,ignoredCounter,profiler->fingerVPTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->fingerVPTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFinger(currObj, query_data, k,profiler->hnswFingerDistanceComputations_VPTree,profiler->fingerVPTreeDistanceCycles,isIdAllowed, profiler);


        profiler->fingerVPTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0_VPTree +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // VPTree - Tri HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnVPTreeTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {

        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (!tri_ready_) 
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }

        tableint currObj = enterpoint_node_;

        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        if (vpt_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;

            const int seed = vpt_->searchEntryPoint(query_f,ignoredCounter,profiler->triVPTreeDistanceCycles,beamWidth,profiler);

            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->triVPTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointTri(currObj, query_data, k,profiler->hnswTriDistanceComputations_VPTree,profiler->triVPTreeDistanceCycles,isIdAllowed, profiler);


        profiler->triVPTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0_VPTree +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }


    // VPTree -> HNSW Normal with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnVPTreeMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;
        if (vpt_ == nullptr)
            return result;
        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get multiple entry points from VPTree
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = vpt_->searchEntryPointMulti(query_f,profiler->vptreeDistanceComputationsMulti,profiler->hnswVPTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->hnswVPTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        // Fallback if no valid VPTree entry points exist
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }


        // Have to comment out next line
        // printMultipleSeedDistances("VPTree-Multi",query_data,entry_points);


        
        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerSTMulti<true>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_VPTreeMulti,profiler->hnswVPTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        else
        {
            top_candidates = searchBaseLayerSTMulti<false>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_VPTreeMulti,profiler->hnswVPTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push({rez.first,getExternalLabel(rez.second)});
            top_candidates.pop();
        }


        profiler->hnswVPTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0VPTreeMulti += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }

    // VPTree -> HNSW Finger with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnVPTreeFingerMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (vpt_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        uint64_t ignoredCounter = 0;
        std::vector<int> seeds = vpt_->searchEntryPointMulti(query_f,ignoredCounter,profiler->fingerVPTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->fingerVPTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 &&
                static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW Finger search using ALL selected entry points


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFingerMulti(entry_points, query_data, k,profiler->hnswFingerDistanceComputations_VPTreeMulti,profiler->fingerVPTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->fingerVPTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0VPTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }

    // VPTree -> HNSW Tri with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnVPTreeTriMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (vpt_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        uint64_t ignoredCounter = 0;
        std::vector<int> seeds = vpt_->searchEntryPointMulti(query_f,ignoredCounter,profiler->triVPTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->triVPTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);

                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW TRI search using ALL selected entry points


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointTriMulti(entry_points, query_data, k,profiler->hnswTriDistanceComputations_VPTreeMulti,profiler->triVPTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->triVPTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0VPTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }



    // PCTree approaches
    // PCTree - Normal HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnPCTree(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {

        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) 
            return result;

        tableint currObj = enterpoint_node_;
        
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        if (pctree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            auto tree_start = std::chrono::high_resolution_clock::now();
            int seed = pctree_->searchNN(query_f,profiler->pctreeDistanceComputations,profiler->hnswPCTreeDistanceCycles,beamWidth,profiler);
            auto tree_end = std::chrono::high_resolution_clock::now();
            profiler->pctreeTimeNs +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->hnswPCTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        // Have to comment out next line
        // printSeedDistance("PCTree",query_data,currObj);

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        if (bare_bone_search) 
        {
            top_candidates = searchBaseLayerST<true, false, true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations_PCTree,profiler->hnswPCTreeDistanceCycles,isIdAllowed, nullptr, profiler);
        } 
        else 
        {
            top_candidates = searchBaseLayerST<false, false, true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations_PCTree,profiler->hnswPCTreeDistanceCycles,isIdAllowed, nullptr, profiler);
        }

        while (top_candidates.size() > k) 
        {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) 
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push({rez.first, getExternalLabel(rez.second)});
            top_candidates.pop();
        }


        profiler->hnswPCTreeLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0_PCTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // PCTree - FINGER HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnPCTreeFinger(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0) 
            return empty;

        tableint currObj = enterpoint_node_;

        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        if (pctree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;
            int seed = pctree_->searchNN(query_f,ignoredCounter,profiler->fingerPCTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->fingerPCTreeUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFinger(currObj, query_data, k,profiler->hnswFingerDistanceComputations_PCTree,profiler->fingerPCTreeDistanceCycles,isIdAllowed, profiler);

        profiler->fingerPCTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0_PCTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }

    // PCTree - Tri HNSW with Single Entry point
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnPCTreeTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {

        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (!tri_ready_) 
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }

        tableint currObj = enterpoint_node_;
        // PCTree finds the starting point
        
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (pctree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;
            int seed = pctree_->searchNN(query_f,ignoredCounter,profiler->triPCTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->triPCTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        // TRI accelerated Level-0 search


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointTri(currObj, query_data, k,profiler->hnswTriDistanceComputations_PCTree,profiler->triPCTreeDistanceCycles,isIdAllowed, profiler);


        profiler->triPCTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0_PCTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }

    // PCTree -> HNSW Normal with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnPCTreeMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;
        if (pctree_ == nullptr)
            return result;
        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get multiple entry points from PCTree
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = pctree_->searchNNMulti(query_f,profiler->pctreeDistanceComputationsMulti,profiler->hnswPCTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->hnswPCTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        // Fallback if no valid PCTree entry points exist
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // have to comment out this line
        // printMultipleSeedDistances("PCTree-Multi",query_data,entry_points);

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerSTMulti<true>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_PCTreeMulti,profiler->hnswPCTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        else
        {
            top_candidates = searchBaseLayerSTMulti<false>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_PCTreeMulti,profiler->hnswPCTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push({rez.first,getExternalLabel(rez.second)});
            top_candidates.pop();
        }

        profiler->hnswPCTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0PCTreeMulti += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // PCTree -> HNSW Finger with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnPCTreeFingerMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (pctree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        uint64_t ignoredCounter=0;
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = pctree_->searchNNMulti(query_f,ignoredCounter,profiler->fingerPCTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->fingerPCTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 &&
                static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW Finger search using ALL selected entry points
        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFingerMulti(entry_points, query_data, k,profiler->hnswFingerDistanceComputations_PCTreeMulti,profiler->fingerPCTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->fingerPCTreeMultiLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0PCTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // PCTree -> HNSW Tri with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnPCTreeTriMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (pctree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        uint64_t ignoredCounter=0;
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = pctree_->searchNNMulti(query_f,ignoredCounter,profiler->triPCTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->triPCTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);

                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW TRI search using ALL selected entry points


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointTriMulti(entry_points, query_data, k,profiler->hnswTriDistanceComputations_PCTreeMulti,profiler->triPCTreeMultiDistanceCycles,isIdAllowed, profiler);


        profiler->triPCTreeMultiLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0PCTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();

        return result;
    }


    // KMeansTree approaches
    // KMeansTree -> HNSW Normal with Single Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTree(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
    
        priority_queue<pair<dist_t, labeltype>> result;

        if (cur_element_count == 0)
            return result;
        tableint currObj = enterpoint_node_;
        
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        if (kmeanstree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            auto tree_start = std::chrono::high_resolution_clock::now();
            int seed = kmeanstree_->searchNN(query_f,profiler->kmeansTreeDistanceComputations,profiler->hnswKMeansTreeDistanceCycles,beamWidth,profiler);
            auto tree_end = std::chrono::high_resolution_clock::now();
            profiler->kmeansTreeTimeNs += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->hnswKMeansTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;


        // have to comment out next line
        // printSeedDistance("KMeansTree",query_data,currObj);



        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search =!num_deleted_ && !isIdAllowed;

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerST<true, false, true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations_KMeansTree,profiler->hnswKMeansTreeDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        else
        {
            top_candidates = searchBaseLayerST<false, false, true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations_KMeansTree,profiler->hnswKMeansTreeDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {   
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push({rez.first,getExternalLabel(rez.second)});
            top_candidates.pop();
        }
        profiler->hnswKMeansTreeLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0_KMeanTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // KMeansTree -> HNSW Finger with Single Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTreeFinger(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        tableint currObj = enterpoint_node_;

        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (kmeanstree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;
            const int seed = kmeanstree_->searchNN(query_f,ignoredCounter,profiler->fingerKMeansTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }
        profiler->fingerKMeansTreeUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        // HNSW Finger search from selected entry point

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFinger(currObj, query_data, k,profiler->hnswFingerDistanceComputations_KMeansTree,profiler->fingerKMeansTreeDistanceCycles,isIdAllowed, profiler);

        profiler->fingerKMeansTreeLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0_KMeanTree += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }
    
    // KMeansTree -> TRI with Single Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTreeTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;

        if (cur_element_count == 0)
            return empty;

        tableint currObj = enterpoint_node_;

        // Get starting point from K-Means Tree
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        if (kmeanstree_ != nullptr)
        {
            const float* query_f = reinterpret_cast<const float*>(query_data);
            uint64_t ignoredCounter = 0;

            const int seed = kmeanstree_->searchNN(query_f,ignoredCounter,profiler->triKMeansTreeDistanceCycles,beamWidth,profiler);
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                currObj = static_cast<tableint>(seed);
            }
        }

        profiler->triKMeansTreeUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;

        // HNSW TRI search from selected entry point

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointTri(currObj, query_data, k,profiler->hnswTriDistanceComputations_KMeansTree,profiler->triKMeansTreeDistanceCycles,isIdAllowed, profiler);

        profiler->triKMeansTreeLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0_KMeanTree +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }


    // KMeansTree -> HNSW Normal with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTreeMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;
        if (kmeanstree_ == nullptr)
            return result;
        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get multiple entry points from KMeansTree
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = kmeanstree_->searchNNMulti(query_f,profiler->kmeansTreeDistanceComputationsMulti,profiler->hnswKMeansTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->hnswKMeansTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        // Fallback if no valid KMeans entry points exist
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // have to comment out next line
        // printMultipleSeedDistances("KMeansTree-Multi",query_data,entry_points);


        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;


        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();
        if (bare_bone_search)
        {
            top_candidates = searchBaseLayerSTMulti<true>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_KMeansTreeMulti,profiler->hnswKMeansTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        else
        {
            top_candidates = searchBaseLayerSTMulti<false>(entry_points, query_data, max(ef_, k),profiler->hnswDistanceComputations_KMeansTreeMulti,profiler->hnswKMeansTreeMultiDistanceCycles,isIdAllowed, nullptr, profiler);
        }
        while (top_candidates.size() > k)
            top_candidates.pop();
        while (!top_candidates.empty())
        {
            pair<dist_t, tableint> rez =top_candidates.top();
            result.push({rez.first,getExternalLabel(rez.second)});
            top_candidates.pop();
        }

        profiler->hnswKMeansTreeMultiLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevel0KMeansTreeMulti += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // KMeansTree -> HNSW Finger with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTreeFingerMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (kmeanstree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
        uint64_t ignoredCounter=0;
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        std::vector<int> seeds = kmeanstree_->searchNNMulti(query_f,ignoredCounter,profiler->fingerKMeansTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->fingerKMeansTreeMultiUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 &&
                static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);
                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW Finger search using ALL selected entry points

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointFingerMulti(entry_points, query_data, k,profiler->hnswFingerDistanceComputations_KMeansTreeMulti,profiler->fingerKMeansTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->fingerKMeansTreeMultiLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0KMeansTreeMulti +=std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }

    // KMeansTree -> HNSW Tri with Multiple Entry Points
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnKMeansTreeTriMulti(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr,int beamWidth = 1) const
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        if (cur_element_count == 0)
            return empty;

        if (kmeanstree_ == nullptr)
            return empty;

        const float* query_f =reinterpret_cast<const float*>(query_data);

        // Get all K representatives from the selected leaf
    
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        uint64_t ignoredCounter = 0;
        std::vector<int> seeds = kmeanstree_->searchNNMulti(query_f,ignoredCounter,profiler->triKMeansTreeMultiDistanceCycles,beamWidth,profiler);
        profiler->triKMeansTreeMultiUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;

        std::vector<tableint> entry_points;
        entry_points.reserve(seeds.size());
        for (int seed : seeds)
        {
            if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
            {
                tableint ep = static_cast<tableint>(seed);

                // Avoid duplicate entry points
                if (std::find(entry_points.begin(),entry_points.end(),ep) == entry_points.end())
                {
                    entry_points.push_back(ep);
                }
            }
        }
        if (entry_points.empty())
        {
            entry_points.push_back(enterpoint_node_);
        }

        // HNSW TRI search using ALL selected entry points

        auto tree_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();


        auto result = searchFromEntryPointTriMulti(entry_points, query_data, k,profiler->hnswTriDistanceComputations_KMeansTreeMulti,profiler->triKMeansTreeMultiDistanceCycles,isIdAllowed, profiler);

        profiler->triKMeansTreeMultiLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto tree_end = std::chrono::high_resolution_clock::now();
        profiler->hnswTriLevel0KMeansTreeMulti += std::chrono::duration<double, std::nano>(tree_end - tree_start).count();


        return result;
    }


    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr) const override
    {
        return searchKnn(query_data, k, isIdAllowed, nullptr);
    }


    // Normal HNSW
    std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const 
    {
        priority_queue<pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) 
            return result;



        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();

        tableint currObj = enterpoint_node_;

        const uint64_t distanceCyclesStart = profiler->readCpuCycles();
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        profiler->hnswDistanceCycles +=profiler->readCpuCycles() - distanceCyclesStart;


        profiler->hnswUpperLayerDistanceComputations++;

        for (int level = maxlevel_; level > 0; level--) 
        {
            auto level_start = std::chrono::high_resolution_clock::now();
            bool changed = true;
            while (changed) 
            {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) 
                {
                    tableint cand = datal[i];

                    if (cand < 0 || cand > max_elements_)
                        throw runtime_error("cand error");


                    const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    profiler->hnswDistanceCycles +=profiler->readCpuCycles() - distanceCyclesStart;

                    profiler->hnswUpperLayerDistanceComputations++;
                    if (d < curdist) 
                    {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }

            auto level_end = std::chrono::high_resolution_clock::now();
            if (profiler != nullptr)
            {
                profiler->hnswLevelTimeNs[level] += std::chrono::duration<double, std::nano>(level_end - level_start).count();
            }
        }

        profiler->hnswUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;

        // have to comment out line
        // printSeedDistance("HNSW-Baseline", query_data, currObj);



        priority_queue<pair<dist_t, tableint>, vector<pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        auto level0_start = std::chrono::high_resolution_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) 
        {
            top_candidates = searchBaseLayerST<true,false,true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations,profiler->hnswDistanceCycles,isIdAllowed,nullptr,profiler);
        } 
        else 
        {
            top_candidates = searchBaseLayerST<false,false,true>(currObj, query_data, max(ef_, k),profiler->hnswDistanceComputations,profiler->hnswDistanceCycles,isIdAllowed,nullptr,profiler);
        }
        while (top_candidates.size() > k) 
        {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) 
        {
            pair<dist_t, tableint> rez = top_candidates.top();
            result.push(pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        profiler->hnswLevel0Cycles += profiler->readCpuCycles() - level0CyclesStart;
        auto level0_end = std::chrono::high_resolution_clock::now();
        profiler->hnswLevelTimeNs[0] += std::chrono::duration<double, std::nano>(level0_end - level0_start).count();
        return result;
    }
    

    // HNSW ->> Finger Optimization
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnFinger(const void *query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const 
    {
        priority_queue<pair<dist_t, labeltype>> result;
        if (cur_element_count == 0)
            return result;


        const uint64_t fingerUpperLayerCyclesStart = profiler->readCpuCycles();
        tableint currObj = enterpoint_node_;

        // Distance Computation
        const uint64_t fingerEntryDistanceCyclesStart = profiler->readCpuCycles();
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        profiler->fingerDistanceCycles +=profiler->readCpuCycles() - fingerEntryDistanceCyclesStart;


        // profiler->hnswFingerDistanceComputations++;
        // Normal HNSW upper-level greedy descent.


        for (int level = maxlevel_; level > 0; --level) 
        {
            bool changed = true;
            while (changed) 
            {
                changed = false;
                unsigned int* data =(unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; ++i) 
                {
                    tableint cand = datal[i];

                    // Distance Computations
                    const uint64_t fingerNeighborDistanceCyclesStart = profiler->readCpuCycles();
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                    profiler->fingerDistanceCycles +=profiler->readCpuCycles() - fingerNeighborDistanceCyclesStart;


                    // profiler->hnswFingerDistanceComputations++;
                    if (d < curdist) 
                    {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        profiler->fingerUpperLayerCycles += profiler->readCpuCycles() - fingerUpperLayerCyclesStart;
        // Level-0 FINGER search.


        auto level0_start =std::chrono::high_resolution_clock::now();
        const uint64_t fingerLevel0CyclesStart = profiler->readCpuCycles();

        result = searchFromEntryPointFinger(currObj,query_data,k,profiler->hnswFingerDistanceComputations,profiler->fingerDistanceCycles,isIdAllowed,profiler);

        profiler->fingerLevel0Cycles += profiler->readCpuCycles() - fingerLevel0CyclesStart;
        auto level0_end = std::chrono::high_resolution_clock::now();
        profiler->hnswFingerLevel0TimeNs +=std::chrono::duration<double, std::nano>(level0_end - level0_start).count();

        return result;
    }


    // HNSW -> TRI optimization
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnnTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const 
    {
        priority_queue<pair<dist_t, labeltype>> empty;
        
        if (cur_element_count == 0)
            return empty;

        if (!tri_ready_) 
        {
            throw runtime_error("TRI search requested before buildTriDistances().");
        }


        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        tableint currObj = enterpoint_node_;
        
        // for distance computation
        const uint64_t triEntryDistanceCyclesStart = profiler->readCpuCycles();
        dist_t curdist =fstdistfunc_(query_data,getDataByInternalId(currObj),dist_func_param_);
        profiler->triDistanceCycles += profiler->readCpuCycles() - triEntryDistanceCyclesStart;
        
        // profiler->hnswTriDistanceComputations++;



        // Normal HNSW upper-level greedy descent.
        for (int level = maxlevel_; level > 0; --level) 
        {
            bool changed = true;
            while (changed) 
            {
                changed = false;
                unsigned int* data =(unsigned int*)get_linklist(currObj, level);
                int size = getListCount(data);
                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; ++i) 
                {
                    tableint cand = datal[i];

                    // for distance computation
                    const uint64_t triEntryDistanceCyclesStart = profiler->readCpuCycles();
                    dist_t d =fstdistfunc_(query_data,getDataByInternalId(cand),dist_func_param_);
                    profiler->triDistanceCycles += profiler->readCpuCycles() - triEntryDistanceCyclesStart;


                    // profiler->hnswTriDistanceComputations++;

                    if (d < curdist) 
                    {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        profiler->triUpperLayerCycles += profiler->readCpuCycles() - upperLayerCyclesStart;



        auto level0_start = std::chrono::high_resolution_clock::now();

        const uint64_t triLevel0CyclesStart = profiler->readCpuCycles();
        auto result = searchFromEntryPointTri(currObj,query_data,k,profiler->hnswTriDistanceComputations,profiler->triDistanceCycles,isIdAllowed,profiler);
        profiler->triLevel0Cycles +=profiler->readCpuCycles() - triLevel0CyclesStart;

        auto level0_end = std::chrono::high_resolution_clock::now();
        if (profiler != nullptr)
        {
            profiler->hnswTriLevel0TimeNs += std::chrono::duration<double, std::nano>(level0_end - level0_start).count();
        }
        return result;
    }


    // HNSW -> TRI optimization at all levels.
    std::priority_queue<std::pair<dist_t, labeltype>>searchKnnTriAllLevels(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr,profilingStats* profiler = nullptr) const
    {
        using ResultQueue =std::priority_queue<std::pair<dist_t, labeltype>>;

        ResultQueue empty;

        if (cur_element_count == 0)
            return empty;

        if (!tri_ready_)
        {
            throw std::runtime_error("TRI search requested before buildTriDistances().");
        }
        
        const uint64_t upperLayerCyclesStart = profiler->readCpuCycles();
        const auto upper_start = std::chrono::steady_clock::now();
        tableint currObj = enterpoint_node_;
        uint64_t distanceCyclesStart = profiler->readCpuCycles();
        dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(currObj), dist_func_param_);
        profiler->hnswTriAllLevelsDistanceCycles +=
        profiler->readCpuCycles() - distanceCyclesStart;

        profiler->hnswUpperLayerDistanceComputationsTRI++;
        

        for (int level = maxlevel_; level > 0; --level)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                tableint currentNode = currObj;
                dist_t currentDist = curdist;
                tableint bestNode = currentNode;
                dist_t bestDist = currentDist;
                linklistsizeint* linklist =get_linklist(currentNode, level);
                size_t neighborCount = getListCount(linklist);
                tableint* neighbors =reinterpret_cast<tableint*>(linklist + 1);

    
                const auto& edgeDistances =upper_tri_edge_distances_[currentNode][level];
                dist_t sqrtQueryDist = std::sqrt(currentDist);
                for (size_t j = 0; j < neighborCount; ++j)
                {
                    tableint candidate = neighbors[j];
                    dist_t edgeDist = edgeDistances[j];
                    
                    dist_t sqrtEdgeDist = std::sqrt(edgeDist);

                    dist_t diff = sqrtQueryDist - sqrtEdgeDist;
                    dist_t lowerBound = diff * diff;

                    if (lowerBound >= bestDist)
                        continue;


                    const uint64_t distanceCyclesStart = profiler->readCpuCycles();
                    dist_t candidateDist = fstdistfunc_(query_data, getDataByInternalId(candidate), dist_func_param_);
                    profiler->hnswTriAllLevelsDistanceCycles +=profiler->readCpuCycles() - distanceCyclesStart;
                    profiler->hnswUpperLayerDistanceComputationsTRI++;
                    
                    if (candidateDist < bestDist)
                    {
                        bestDist = candidateDist;
                        bestNode = candidate;
                    }
                }

                if (bestNode != currentNode)
                {
                    currObj = bestNode;
                    curdist = bestDist;
                    changed = true;
                }
            }
        }

        // printSeedDistance("HNSW-TRI-AllLevels", query_data, currObj);
        profiler->hnswTriAllLevelsUpperLayerCycles +=profiler->readCpuCycles() - upperLayerCyclesStart;

        const auto upper_end = std::chrono::steady_clock::now();

        if (profiler != nullptr)
        {
            profiler->hnswTriAllLevelsUpperLayerTimeNs += std::chrono::duration<double, std::nano>(upper_end - upper_start).count();
        }

        const auto level0_start = std::chrono::steady_clock::now();
        const uint64_t level0CyclesStart = profiler->readCpuCycles();

        auto result = searchFromEntryPointTri(currObj, query_data, k,profiler->hnswLevel0DistanceComputationsWithTRI,profiler->hnswTriAllLevelsDistanceCycles,isIdAllowed, profiler);

        profiler->hnswTriAllLevelsLevel0Cycles +=profiler->readCpuCycles() - level0CyclesStart;
        const auto level0_end = std::chrono::steady_clock::now();
        profiler->hnswTriAllLevelsLevel0TimeNs +=std::chrono::duration<double, std::nano>(level0_end - level0_start).count();

        return result;
    }
    

    
    // VTree -> TRI
    // std::priority_queue<std::pair<dist_t, labeltype>>
    // searchKnnVTreeTri(const void* query_data,size_t k,BaseFilterFunctor* isIdAllowed = nullptr) const 
    // {
    //     priority_queue<pair<dist_t, labeltype>> empty;
    //     if (cur_element_count == 0)
    //         return empty;

    //     if (!tri_ready_) 
    //     {
    //         throw runtime_error(
    //             "TRI search requested before buildTriDistances().");
    //     }
    //     tableint currObj = enterpoint_node_;
    //     dist_t curdist =fstdistfunc_(query_data,getDataByInternalId(currObj),dist_func_param_);
    //     if (vtree_ != nullptr) 
    //     {
    //         const float* query_f =reinterpret_cast<const float*>(query_data);
    //         const int seed = vtree_->searchEntryPoint(query_f);
    //         if (seed >= 0 && static_cast<size_t>(seed) < cur_element_count)
    //         {
    //             dist_t d = fstdistfunc_(query_data,getDataByInternalId(static_cast<tableint>(seed)),dist_func_param_);

    //             if (d < curdist)
    //             {
    //                 curdist = d;
    //                 currObj = static_cast<tableint>(seed);
    //             }
    //         }
    //     }
    //     return searchFromEntryPointTri(currObj,query_data,k,isIdAllowed);
    // }



    vector<pair<dist_t, labeltype >>
    searchStopConditionClosest(const void *query_data,BaseSearchStopCondition<dist_t>& stop_condition,BaseFilterFunctor* isIdAllowed = nullptr) const 
    {
        vector<pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) 
            return result;
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
        for (int level = maxlevel_; level > 0; level--) 
        {
            bool changed = true;
            while (changed) 
            {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) 
                {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) 
                    {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        priority_queue<pair<dist_t, tableint>,vector<pair<dist_t, tableint>>,CompareByFirst> top_candidates;
        uint64_t ignoredCounter = 0;
        uint64_t ignoredDistanceCycles = 0;top_candidates = searchBaseLayerST<false>(currObj, query_data, 0,ignoredCounter, ignoredDistanceCycles,isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) 
        {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }

    void checkIntegrity() 
    {
        int connections_checked = 0;
        vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) 
        {
            for (int l = 0; l <= element_levels_[i]; l++) 
            {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                unordered_set<tableint> s;
                for (int j = 0; j < size; j++) 
                {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) 
        {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) 
            {
                assert(inbound_connections_num[i] > 0);
                min1 = min(inbound_connections_num[i], min1);
                max1 = max(inbound_connections_num[i], max1);
            }
            cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};

}