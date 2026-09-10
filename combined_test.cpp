#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include <limits>
#include "hnswlib/hnswlib.h"

std::vector<float> read_fvecs(const std::string& path, int& dim, int& n) 
{
    std::ifstream f(path, std::ios::binary);
    if (!f) 
        throw std::runtime_error("Cannot open " + path);

    int32_t d;
    f.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    f.seekg(0, std::ios::end);
    std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    const std::streamsize record =4 + static_cast<std::streamsize>(d) * 4;

    n = static_cast<int>(fsize / record);

    std::vector<float> data(static_cast<size_t>(n) * d);

    for (int i = 0; i < n; ++i) 
    {
        int32_t dd;
        f.read(reinterpret_cast<char*>(&dd), 4);
        if (dd != d)
            throw std::runtime_error("Inconsistent fvecs dimension");

        f.read(reinterpret_cast<char*>(&data[static_cast<size_t>(i) * d]),4 * d);
    }
    return data;
}

std::vector<int> read_ivecs(const std::string& path,int& dim,int& n) 
{

    std::ifstream f(path, std::ios::binary);
    if (!f) 
        throw std::runtime_error("Cannot open " + path);

    int32_t d;
    f.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    f.seekg(0, std::ios::end);
    std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    const std::streamsize record =4 + static_cast<std::streamsize>(d) * 4;

    n = static_cast<int>(fsize / record);

    std::vector<int> data(static_cast<size_t>(n) * d);

    for (int i = 0; i < n; ++i) 
    {
        int32_t dd;
        f.read(reinterpret_cast<char*>(&dd), 4);
        if (dd != d)
            throw std::runtime_error("Inconsistent ivecs dimension");

        f.read(reinterpret_cast<char*>(&data[static_cast<size_t>(i) * d]),4 * d);
    }
    return data;
}

float recall_at_k(const int* gt,int gt_k,const std::vector<size_t>& returned,int k) 
{
    const int effective_k = std::min(gt_k, k);
    std::unordered_set<int> gt_set(gt,gt + effective_k);
    int hits = 0;
    for (size_t id : returned) 
    {
        if (gt_set.count(static_cast<int>(id)))
            ++hits;
    }

    return static_cast<float>(hits) /static_cast<float>(effective_k);
}

std::vector<size_t> extract_ids(std::priority_queue<std::pair<float, hnswlib::labeltype>>& pq,int k) 
{
    std::vector<size_t> ids;
    ids.reserve(k);
    while (!pq.empty()) 
    {
        ids.push_back(static_cast<size_t>(pq.top().second));
        pq.pop();
    }
    return ids;
}

int main(int argc, char** argv) 
{
    const std::string data_dir =(argc > 1)? argv[1]: "./datasets/sift";
    const std::string base_path =data_dir + "/base.fvecs";
    const std::string query_path =data_dir + "/query.fvecs";
    const std::string gt_path =data_dir + "/groundtruth.ivecs";



    // Parameters
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 50;
    const int k = 30;
    const int num_queries = 10000;
    const int leaf_capacity = 64;
    const int finger_rank = 16;
    const int finger_warmup = 8;
    // const int vt_pivots = 2;
    const int pctree_partitions = 4;
    const int mtree_pivots=5;
    const int kmeans_clusters = 4;
    const int kmeans_leaf_clusters = 3;
    const int pctree_leaf_clusters = 3;
    const int mtree_leaf_clusters = 3;
    const int vpt_leaf_clusters = 3;



    // Loading dataset base_vectors , ground Truth and queries

    std::cout<< "Loading dataset base vectors from "<< base_path << " ...\n";
    int base_dim, base_n;
    auto base =read_fvecs(base_path,base_dim,base_n);
    std::cout<< "  base: "<< base_n<< " x "<< base_dim<< "\n";
    std::cout<< "Loading queries from "<< query_path<< " ...\n";
    int q_dim, q_n;
    auto queries =read_fvecs(query_path,q_dim,q_n);
    std::cout<<"queries: "<< q_n<< " x "<< q_dim<< "\n";
    if (base_dim != q_dim)
        throw std::runtime_error("Dimension mismatch between base and queries");
    std::cout<< "Loading ground truth from "<< gt_path<< " ...\n";
    int gt_dim, gt_n;
    auto gt =read_ivecs(gt_path,gt_dim,gt_n);
    std::cout<<"  gt: "<< gt_n<< " x "<< gt_dim<< "\n\n";
    const int dim = base_dim;
    const int N = base_n;
    const int Q = std::min(num_queries, q_n);



    // Bulding HNSW Graph
    std::cout<< "Building HNSW index over "<< N<< " vectors (M="<< M<< ", ef_construction="<< ef_construction<< ")...\n";
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space,N,M,ef_construction);
    auto t_build_start =std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) 
    {
        hnsw.addPoint(base.data() +static_cast<size_t>(i) * dim,static_cast<size_t>(i));
        if ((i + 1) % 100000 == 0) 
        {
            std::cout<< "  indexed "<< (i + 1)<< " / "<< N<< "\n";
        }
    }
    auto t_build_end =std::chrono::high_resolution_clock::now();
    double build_sec =std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout<< "HNSW built in "<< std::fixed<< std::setprecision(1)<< build_sec<< " s. Max level = "<< hnsw.maxlevel_<< "\n";
    hnsw.setEf(ef_search);
    const int hnsw_num_levels = hnsw.maxlevel_ + 1;


    // Building PCTree
    std::cout<< "\nBuilding PCTree "<< "(leafCapacity="<< leaf_capacity<< ", numPartitions="<< pctree_partitions<< ", leafClusters="<< pctree_leaf_clusters<< ")...\n";
    auto t_pc_start =std::chrono::high_resolution_clock::now();
    hnsw.buildPCTree(pctree_partitions,leaf_capacity,pctree_leaf_clusters);
    auto t_pc_end =std::chrono::high_resolution_clock::now();
    double pc_sec =std::chrono::duration<double>(t_pc_end - t_pc_start).count();
    std::cout<< "PCTree built in "<< std::setprecision(2)<< pc_sec<< " s.\n";
    const int pctree_height = hnsw.getPCTreeHeight();
    std::cout<< "PCTree height: "<< pctree_height<< "\n";



    // Building MTree
    std::cout<< "\nBuilding MTree "<< "(leafCapacity="<< leaf_capacity<< ", leafClusters="<< mtree_leaf_clusters<< ")...\n";
    auto t_mtree_start =std::chrono::high_resolution_clock::now();
    hnsw.buildMTree(mtree_pivots, leaf_capacity, mtree_leaf_clusters);
    auto t_mtree_end =std::chrono::high_resolution_clock::now();
    double mtree_sec =std::chrono::duration<double>(t_mtree_end - t_mtree_start).count();
    std::cout<< "MTree built in "<< std::setprecision(2)<< mtree_sec<< " s.\n";
    const int mtree_height = hnsw.getMTreeHeight();
    std::cout<< "MTree height: "<< mtree_height<< "\n";




    // Building VPTree
    std::cout<< "\nBuilding VantagePointTree "<< "(leafCapacity="<< leaf_capacity<< ", leafClusters="<< vpt_leaf_clusters<< ")...\n";
    auto t_vp_start =std::chrono::high_resolution_clock::now();
    hnsw.buildVantagePointTree(leaf_capacity, vpt_leaf_clusters);
    auto t_vp_end =std::chrono::high_resolution_clock::now();
    double vpt_build_time =std::chrono::duration<double>(t_vp_end - t_vp_start).count();
    std::cout<< "VantagePointTree built in "<< std::fixed<< std::setprecision(2)<< vpt_build_time<< " s.\n";
    const int vptree_height = hnsw.getVPTreeHeight();
    std::cout<< "VantagePointTree height: "<< vptree_height << "\n";




    // Skipping VTree because it is not performing good on higher dimension
    // Building VTree
    // std::cout<< "\nBuilding VoronoiTree "<< "(numPivots="<< vt_pivots<< ", leafCapacity="<< leaf_capacity<< ")...\n";
    // auto start =std::chrono::high_resolution_clock::now();
    // hnsw.buildVoronoiTree(vt_pivots, leaf_capacity);
    // auto end =std::chrono::high_resolution_clock::now();
    // double vtree_build_time =std::chrono::duration<double>(end - start).count();
    // std::cout<< "VoronoiTree built in "<< std::fixed<< std::setprecision(2)<< vtree_build_time<< " s.\n";
    // const int vtree_height = hnsw.getVTreeHeight();
    // std::cout<< "VoronoiTree height: "<< vtree_height << "\n";



    // Building KMeansTree
    std::cout << "\nBuilding KMeansTree "<< "(K=" << kmeans_clusters<< ", leafCapacity=" << leaf_capacity << ")...\n";
    auto t_kmeans_start = std::chrono::high_resolution_clock::now();
    hnsw.buildKMeansTree(kmeans_clusters, leaf_capacity,kmeans_leaf_clusters);
    auto t_kmeans_end = std::chrono::high_resolution_clock::now();
    double kmeans_build_sec =std::chrono::duration<double>(t_kmeans_end - t_kmeans_start).count();
    std::cout << "KMeansTree built in "<< std::fixed << std::setprecision(2)<< kmeans_build_sec << " s.\n";
    const int kmeans_height = hnsw.getKMeansTreeHeight();
    std::cout << "KMeansTree height: "<< kmeans_height << "\n";


    // Finger Optimization preprocessing
    std::cout<< "\nBuilding FINGER structures...\n";
    hnsw.setFingerRank(finger_rank);
    hnsw.setFingerWarmup(finger_warmup);
    hnsw.setUseFinger(true, finger_rank);
    auto t_finger_start =std::chrono::high_resolution_clock::now();
    hnsw.buildFingerProjections();
    auto t_finger_end =std::chrono::high_resolution_clock::now();
    double finger_build_sec =std::chrono::duration<double>(t_finger_end - t_finger_start).count();
    std::cout<< "FINGER built in "<< std::fixed<< std::setprecision(2)<< finger_build_sec<< " s.\n";
    std::cout<< "FINGER rank: "<< hnsw.getFingerRank()<< "\n";
    std::cout<< "FINGER warmup: "<< hnsw.getFingerWarmup()<< "\n\n";

    // TRI-Scheme preprocessing
    std::cout << "Building TRI-Scheme structures...\n";
    auto t_tri_start = std::chrono::high_resolution_clock::now();
    hnsw.buildTriDistances();
    auto t_tri_end = std::chrono::high_resolution_clock::now();
    double tri_build_sec =
        std::chrono::duration<double>(t_tri_end - t_tri_start).count();
    std::cout << "TRI structures built in " << std::fixed << std::setprecision(2)
              << tri_build_sec << " s.\n\n";

    std::cout<< "Running "<< Q<< " queries (k="<< k<< ", ef="<< ef_search<< ")...\n\n";



    double total_recall_hnsw = 0.0;
    double total_recall_hnsw_finger = 0.0;
    double total_recall_hnsw_tri = 0.0;
    double total_recall_pctree = 0.0;
    double total_recall_pctree_finger = 0.0;
    double total_recall_pctree_tri = 0.0;
    double total_recall_mtree_tri = 0.0;
    double total_recall_mtree = 0.0;
    double total_recall_mtree_finger = 0.0;
    // double total_recall_vtree_tri = 0.0;
    // double total_recall_vtree = 0.0;
    // double total_recall_vtree_finger = 0.0;
    double total_recall_vpt_tri = 0.0;
    double total_recall_vpt=0.0;
    double total_recall_vpt_finger = 0.0;
    double total_recall_kmeans = 0.0;
    double total_recall_kmeans_finger = 0.0;
    double total_recall_kmeans_tri = 0.0;
    double total_recall_kmeans_multi = 0.0;
    double total_recall_kmeans_finger_multi = 0.0;
    double total_recall_kmeans_tri_multi = 0.0;
    double total_recall_pctree_multi = 0.0;
    double total_recall_pctree_finger_multi = 0.0;
    double total_recall_pctree_tri_multi = 0.0;
    double total_recall_mtree_multi = 0.0;
    double total_recall_mtree_finger_multi = 0.0;
    double total_recall_mtree_tri_multi = 0.0;
    double total_recall_vpt_multi = 0.0;
    double total_recall_vpt_finger_multi = 0.0;
    double total_recall_vpt_tri_multi = 0.0;



    // Accumulate nanoseconds as double so sub-microsecond measurements are not
    // truncated to zero before averaging.
    double time_hnsw_ns = 0.0;
    double time_hnsw_finger_ns = 0.0;
    double time_hnsw_tri_ns = 0.0;
    double time_pctree_ns = 0.0;
    double time_pctree_finger_ns = 0.0;
    double time_pctree_tri_ns = 0.0;
    double time_mtree_tri_ns = 0.0;
    double time_mtree_ns = 0.0;
    double time_mtree_finger_ns = 0.0;
    // double time_vtree_tri_ns = 0.0;
    // double time_vtree_ns = 0.0;
    // double time_vtree_finger_ns = 0.0;
    double time_vpt_tri_ns = 0.0;
    double time_vpt_ns = 0.0;
    double time_vpt_finger_ns = 0.0;
    double time_kmeans_ns = 0.0;
    double time_kmeans_finger_ns = 0.0;
    double time_kmeans_tri_ns = 0.0;
    double time_kmeans_multi_ns = 0.0;
    double time_kmeans_finger_multi_ns = 0.0;
    double time_kmeans_tri_multi_ns = 0.0;
    double time_pctree_multi_ns = 0.0;
    double time_pctree_finger_multi_ns = 0.0;
    double time_pctree_tri_multi_ns = 0.0;
    double time_mtree_multi_ns = 0.0;
    double time_mtree_finger_multi_ns = 0.0;
    double time_mtree_tri_multi_ns = 0.0;
    double time_vpt_multi_ns = 0.0;
    double time_vpt_finger_multi_ns = 0.0;
    double time_vpt_tri_multi_ns = 0.0;



    for (int q = 0; q < Q; ++q) 
    {
        const float* query =queries.data() +static_cast<size_t>(q) * dim;
        const int* gt_q =gt.data() +static_cast<size_t>(q) * gt_dim;


        // for baseline hnsw
        auto start =std::chrono::high_resolution_clock::now();
        auto pq_hnsw =hnsw.searchKnn(query,static_cast<size_t>(k));
        auto end =std::chrono::high_resolution_clock::now();
        time_hnsw_ns += std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> hnsw_ids =extract_ids(pq_hnsw, k);
        total_recall_hnsw +=recall_at_k(gt_q,gt_dim,hnsw_ids,k);


        // HNSW -> TRI
        start = std::chrono::high_resolution_clock::now();
        auto pq_hnsw_tri = hnsw.searchKnnTri(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_hnsw_tri_ns += std::chrono::duration<double, std::nano>(end-start).count();
        auto hnsw_tri_ids = extract_ids(pq_hnsw_tri, k);
        total_recall_hnsw_tri += recall_at_k(gt_q, gt_dim, hnsw_tri_ids, k);


        // HNSW + FINGER
        start = std::chrono::high_resolution_clock::now();
        auto pq_hnsw_finger =hnsw.searchKnnFinger(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_hnsw_finger_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> hnsw_finger_ids =extract_ids(pq_hnsw_finger, k);
        total_recall_hnsw_finger +=
        recall_at_k(gt_q, gt_dim, hnsw_finger_ids, k);


        // for mtree->hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_mtree =hnsw.searchKnnMTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_mtree_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_ids =extract_ids(pq_mtree,k);
        total_recall_mtree +=recall_at_k(gt_q,gt_dim,mtree_ids,k);


        // MTree -> TRI
        start = std::chrono::high_resolution_clock::now();
        auto pq_mtree_tri = hnsw.searchKnnMTreeTri(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_mtree_tri_ns += std::chrono::duration<double, std::nano>(end-start).count();
        auto mtree_tri_ids = extract_ids(pq_mtree_tri, k);
        total_recall_mtree_tri += recall_at_k(gt_q, gt_dim, mtree_tri_ids, k);


        // for mtree->(level-0 + finer optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_mtree_finger =hnsw.searchKnnMTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_mtree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_finger_ids =extract_ids(pq_mtree_finger,k);
        total_recall_mtree_finger +=recall_at_k(gt_q,gt_dim,mtree_finger_ids,k);


        // MTree -> HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_mtree_multi =hnsw.searchKnnMTreeMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_mtree_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_multi_ids = extract_ids(pq_mtree_multi, k);
        total_recall_mtree_multi +=recall_at_k(gt_q, gt_dim, mtree_multi_ids, k);


        // MTree -> TRI HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_mtree_tri_multi =hnsw.searchKnnMTreeTriMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_mtree_tri_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_tri_multi_ids = extract_ids(pq_mtree_tri_multi, k);
        total_recall_mtree_tri_multi +=recall_at_k(gt_q, gt_dim, mtree_tri_multi_ids, k);


        // MTree -> FINGER HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_mtree_finger_multi =hnsw.searchKnnMTreeFingerMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_mtree_finger_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_finger_multi_ids =extract_ids(pq_mtree_finger_multi, k);
        total_recall_mtree_finger_multi +=recall_at_k(gt_q, gt_dim, mtree_finger_multi_ids, k);


        // for pctree -> hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_pctree =hnsw.searchKnnPCTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_pctree_ns += std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> pctree_ids =extract_ids(pq_pctree, k);
        total_recall_pctree +=recall_at_k(gt_q,gt_dim,pctree_ids,k);


        // PCTree -> TRI
        start = std::chrono::high_resolution_clock::now();
        auto pq_pctree_tri = hnsw.searchKnnPCTreeTri(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_pctree_tri_ns += std::chrono::duration<double, std::nano>(end-start).count();
        auto pctree_tri_ids = extract_ids(pq_pctree_tri, k);
        total_recall_pctree_tri += recall_at_k(gt_q, gt_dim, pctree_tri_ids, k);


        // for pctree -> hnsw(level-0 + finger optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_pctree_finger =hnsw.searchKnnPCTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_pctree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> pctree_finger_ids =extract_ids(pq_pctree_finger,k);
        total_recall_pctree_finger +=recall_at_k(gt_q,gt_dim,pctree_finger_ids,k);


        // PCTree -> HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_pctree_multi =hnsw.searchKnnPCTreeMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_pctree_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto pctree_multi_ids = extract_ids(pq_pctree_multi, k);
        total_recall_pctree_multi +=recall_at_k(gt_q, gt_dim, pctree_multi_ids, k);


        // PCTree -> TRI HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_pctree_tri_multi =hnsw.searchKnnPCTreeTriMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_pctree_tri_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto pctree_tri_multi_ids = extract_ids(pq_pctree_tri_multi, k);
        total_recall_pctree_tri_multi +=recall_at_k(gt_q, gt_dim, pctree_tri_multi_ids, k);


        // PCTree -> FINGER HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_pctree_finger_multi =hnsw.searchKnnPCTreeFingerMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_pctree_finger_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto pctree_finger_multi_ids =extract_ids(pq_pctree_finger_multi, k);
        total_recall_pctree_finger_multi +=recall_at_k(gt_q, gt_dim, pctree_finger_multi_ids, k);


        // for vptree->hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vpt =hnsw.searchKnnVPTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vpt_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_ids =extract_ids(pq_vpt, k);
        total_recall_vpt +=recall_at_k(gt_q,gt_dim,vpt_ids,k);


        // VPTree -> TRI
        start = std::chrono::high_resolution_clock::now();
        auto pq_vpt_tri = hnsw.searchKnnVPTreeTri(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_vpt_tri_ns += std::chrono::duration<double, std::nano>(end-start).count();
        auto vpt_tri_ids = extract_ids(pq_vpt_tri, k);
        total_recall_vpt_tri += recall_at_k(gt_q, gt_dim, vpt_tri_ids, k);


        // for vptree->(level-0 + finer optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vpt_finger =hnsw.searchKnnVPTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vpt_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_finger_ids =extract_ids(pq_vpt_finger, k);
        total_recall_vpt_finger +=recall_at_k(gt_q,gt_dim,vpt_finger_ids,k);


        // VPTree -> HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_vpt_multi =hnsw.searchKnnVPTreeMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_vpt_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_multi_ids = extract_ids(pq_vpt_multi, k);
        total_recall_vpt_multi +=recall_at_k(gt_q, gt_dim, vpt_multi_ids, k);


        // VPTree -> TRI HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_vpt_tri_multi =hnsw.searchKnnVPTreeTriMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_vpt_tri_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_tri_multi_ids = extract_ids(pq_vpt_tri_multi, k);
        total_recall_vpt_tri_multi +=recall_at_k(gt_q, gt_dim, vpt_tri_multi_ids, k);


        // VPTree -> FINGER HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_vpt_finger_multi =hnsw.searchKnnVPTreeFingerMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_vpt_finger_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_finger_multi_ids =extract_ids(pq_vpt_finger_multi, k);
        total_recall_vpt_finger_multi +=recall_at_k(gt_q, gt_dim, vpt_finger_multi_ids, k);



        // VTree is not performing good for higher dimension

        // for vtree->hnsw(level-0)
        // start =std::chrono::high_resolution_clock::now();
        // auto pq_vtree =hnsw.searchKnnVTree(query,static_cast<size_t>(k));
        // end =std::chrono::high_resolution_clock::now();
        // time_vtree_ns += std::chrono::duration<double, std::nano>(end - start).count();
        // auto vtree_ids =extract_ids(pq_vtree, k);
        // total_recall_vtree +=recall_at_k(gt_q,gt_dim,vtree_ids,k);


        // VTree -> TRI
        // start = std::chrono::high_resolution_clock::now();
        // auto pq_vtree_tri = hnsw.searchKnnVTreeTri(query, static_cast<size_t>(k));
        // end = std::chrono::high_resolution_clock::now();
        // time_vtree_tri_ns += std::chrono::duration<double, std::nano>(end-start).count();
        // auto vtree_tri_ids = extract_ids(pq_vtree_tri, k);
        // total_recall_vtree_tri += recall_at_k(gt_q, gt_dim, vtree_tri_ids, k);


        // for vtree->(level-0 + finer optimization)
        // start =std::chrono::high_resolution_clock::now();
        // auto pq_vtree_finger =hnsw.searchKnnVTreeFinger(query,static_cast<size_t>(k));
        // end =std::chrono::high_resolution_clock::now();
        // time_vtree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        // auto vtree_finger_ids =extract_ids(pq_vtree_finger,k);
        // total_recall_vtree_finger +=recall_at_k(gt_q,gt_dim,vtree_finger_ids,k);
        

        // KMeansTree -> HNSW
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans =hnsw.searchKnnKMeansTree(query,static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_ids =extract_ids(pq_kmeans, k);
        total_recall_kmeans +=recall_at_k(gt_q,gt_dim,kmeans_ids,k);


        // KMeansTree -> TRI HNSW
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans_tri =hnsw.searchKnnKMeansTreeTri(query,static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_tri_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_tri_ids =extract_ids(pq_kmeans_tri, k);
        total_recall_kmeans_tri +=recall_at_k(gt_q,gt_dim,kmeans_tri_ids,k);


        // KMeansTree -> finger HNSW
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans_finger =hnsw.searchKnnKMeansTreeFinger(query,static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_finger_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_finger_ids =extract_ids(pq_kmeans_finger, k);
        total_recall_kmeans_finger +=recall_at_k(gt_q,gt_dim,kmeans_finger_ids,k);


        // KMeansTree -> HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans_multi =hnsw.searchKnnKMeansTreeMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_multi_ids = extract_ids(pq_kmeans_multi, k);
        total_recall_kmeans_multi +=recall_at_k(gt_q, gt_dim, kmeans_multi_ids, k);

        // KMeansTree -> TRI HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans_tri_multi =hnsw.searchKnnKMeansTreeTriMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_tri_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_tri_multi_ids = extract_ids(pq_kmeans_tri_multi, k);
        total_recall_kmeans_tri_multi +=recall_at_k(gt_q, gt_dim, kmeans_tri_multi_ids, k);


        // KMeansTree -> FINGER HNSW Multi
        start = std::chrono::high_resolution_clock::now();
        auto pq_kmeans_finger_multi =hnsw.searchKnnKMeansTreeFingerMulti(query, static_cast<size_t>(k));
        end = std::chrono::high_resolution_clock::now();
        time_kmeans_finger_multi_ns +=std::chrono::duration<double, std::nano>(end - start).count();
        auto kmeans_finger_multi_ids =extract_ids(pq_kmeans_finger_multi, k);
        total_recall_kmeans_finger_multi +=recall_at_k(gt_q, gt_dim, kmeans_finger_multi_ids, k);


        if ((q + 1) % 1000 == 0) 
        {
            std::cout<< "  completed "<< (q + 1)<< " / "<< Q<< "\n";
        }
    }

    const double avg_recall_hnsw = total_recall_hnsw / Q;
    const double avg_recall_hnsw_finger =total_recall_hnsw_finger / Q;
    const double avg_recall_hnsw_tri = total_recall_hnsw_tri / Q;
    const double avg_recall_pctree = total_recall_pctree / Q;
    const double avg_recall_pctree_finger = total_recall_pctree_finger / Q;
    const double avg_recall_pctree_tri = total_recall_pctree_tri / Q;
    const double avg_recall_mtree_tri = total_recall_mtree_tri / Q;
    const double avg_recall_mtree = total_recall_mtree / Q;
    const double avg_recall_mtree_finger = total_recall_mtree_finger / Q;
    // const double avg_recall_vtree_tri = total_recall_vtree_tri / Q;
    // const double avg_recall_vtree = total_recall_vtree / Q;
    // const double avg_recall_vtree_finger = total_recall_vtree_finger / Q;
    const double avg_recall_vpt_tri = total_recall_vpt_tri / Q;
    const double avg_recall_vpt = total_recall_vpt / Q;
    const double avg_recall_vpt_finger = total_recall_vpt_finger / Q;
    const double avg_recall_kmeans =total_recall_kmeans / Q;
    const double avg_recall_kmeans_finger =total_recall_kmeans_finger / Q;
    const double avg_recall_kmeans_tri =total_recall_kmeans_tri / Q;
    const double avg_recall_kmeans_multi = total_recall_kmeans_multi / Q;
    const double avg_recall_kmeans_finger_multi =total_recall_kmeans_finger_multi / Q;
    const double avg_recall_kmeans_tri_multi = total_recall_kmeans_tri_multi / Q;
    const double avg_recall_pctree_multi = total_recall_pctree_multi / Q;
    const double avg_recall_pctree_finger_multi = total_recall_pctree_finger_multi / Q;
    const double avg_recall_pctree_tri_multi = total_recall_pctree_tri_multi / Q;
    const double avg_recall_mtree_multi = total_recall_mtree_multi / Q;
    const double avg_recall_mtree_finger_multi = total_recall_mtree_finger_multi / Q;
    const double avg_recall_mtree_tri_multi = total_recall_mtree_tri_multi / Q;
    const double avg_recall_vpt_multi = total_recall_vpt_multi / Q;
    const double avg_recall_vpt_finger_multi = total_recall_vpt_finger_multi / Q;
    const double avg_recall_vpt_tri_multi = total_recall_vpt_tri_multi / Q;



    const double avg_time_hnsw_us = (time_hnsw_ns / Q) / 1000.0;
    const double avg_time_hnsw_finger_us = (time_hnsw_finger_ns / Q) / 1000.0;
    const double avg_time_pctree_us = (time_pctree_ns / Q) / 1000.0;
    const double avg_time_pctree_finger_us = (time_pctree_finger_ns / Q) / 1000.0;
    const double avg_time_pctree_tri_us = (time_pctree_tri_ns / Q) / 1000.0;
    const double avg_time_hnsw_tri_us = (time_hnsw_tri_ns / Q) / 1000.0;
    const double avg_time_mtree_tri_us = (time_mtree_tri_ns / Q) / 1000.0;
    const double avg_time_mtree_us = (time_mtree_ns / Q) / 1000.0;
    const double avg_time_mtree_finger_us = (time_mtree_finger_ns / Q) / 1000.0;
    // const double avg_time_vtree_tri_us = (time_vtree_tri_ns / Q) / 1000.0;
    // const double avg_time_vtree_us = (time_vtree_ns / Q) / 1000.0;
    // const double avg_time_vtree_finger_us = (time_vtree_finger_ns / Q) / 1000.0;
    const double avg_time_vpt_tri_us = (time_vpt_tri_ns / Q) / 1000.0;
    const double avg_time_vpt_us = (time_vpt_ns / Q) / 1000.0;
    const double avg_time_vpt_finger_us = (time_vpt_finger_ns / Q) / 1000.0;
    const double avg_time_kmeans_us =(time_kmeans_ns / Q) / 1000.0;
    const double avg_time_kmeans_finger_us =(time_kmeans_finger_ns / Q) / 1000.0;
    const double avg_time_kmeans_tri_us =(time_kmeans_tri_ns / Q) / 1000.0;
    const double avg_time_kmeans_multi_us = (time_kmeans_multi_ns / Q) / 1000.0;
    const double avg_time_kmeans_finger_multi_us = (time_kmeans_finger_multi_ns / Q) / 1000.0;
    const double avg_time_kmeans_tri_multi_us = (time_kmeans_tri_multi_ns / Q) / 1000.0;
    const double avg_time_pctree_multi_us = (time_pctree_multi_ns / Q) / 1000.0;
    const double avg_time_pctree_finger_multi_us = (time_pctree_finger_multi_ns / Q) / 1000.0;
    const double avg_time_pctree_tri_multi_us = (time_pctree_tri_multi_ns / Q) / 1000.0;
    const double avg_time_mtree_multi_us = (time_mtree_multi_ns / Q) / 1000.0;
    const double avg_time_mtree_finger_multi_us = (time_mtree_finger_multi_ns / Q) / 1000.0;
    const double avg_time_mtree_tri_multi_us = (time_mtree_tri_multi_ns / Q) / 1000.0;
    const double avg_time_vpt_multi_us = (time_vpt_multi_ns / Q) / 1000.0;
    const double avg_time_vpt_finger_multi_us = (time_vpt_finger_multi_ns / Q) / 1000.0;
    const double avg_time_vpt_tri_multi_us = (time_vpt_tri_multi_ns / Q) / 1000.0;



    const double speedup_pctree = avg_time_hnsw_us / avg_time_pctree_us;
    const double speedup_hnsw_finger = avg_time_hnsw_us / avg_time_hnsw_finger_us;
    const double speedup_pctree_finger = avg_time_hnsw_us / avg_time_pctree_finger_us;
    const double speedup_pctree_tri = avg_time_hnsw_us / avg_time_pctree_tri_us;
    const double speedup_hnsw_tri = avg_time_hnsw_us / avg_time_hnsw_tri_us;
    const double speedup_mtree_tri = avg_time_hnsw_us / avg_time_mtree_tri_us;
    const double speedup_mtree = avg_time_hnsw_us / avg_time_mtree_us;
    const double speedup_mtree_finger = avg_time_hnsw_us / avg_time_mtree_finger_us;
    // const double speedup_vtree_tri = avg_time_hnsw_us / avg_time_vtree_tri_us;
    // const double speedup_vtree = avg_time_hnsw_us / avg_time_vtree_us;
    // const double speedup_vtree_finger = avg_time_hnsw_us / avg_time_vtree_finger_us;
    const double speedup_vpt_tri = avg_time_hnsw_us / avg_time_vpt_tri_us;
    const double speedup_vpt = avg_time_hnsw_us / avg_time_vpt_us;
    const double speedup_vpt_finger = avg_time_hnsw_us / avg_time_vpt_finger_us;
    const double speedup_kmeans =avg_time_hnsw_us / avg_time_kmeans_us;
    const double speedup_kmeans_finger =avg_time_hnsw_us / avg_time_kmeans_finger_us;
    const double speedup_kmeans_tri =avg_time_hnsw_us / avg_time_kmeans_tri_us;
    const double speedup_kmeans_multi =avg_time_hnsw_us / avg_time_kmeans_multi_us;
    const double speedup_kmeans_finger_multi =avg_time_hnsw_us / avg_time_kmeans_finger_multi_us;
    const double speedup_kmeans_tri_multi = avg_time_hnsw_us / avg_time_kmeans_tri_multi_us;
    const double speedup_pctree_multi = avg_time_hnsw_us / avg_time_pctree_multi_us;
    const double speedup_pctree_finger_multi = avg_time_hnsw_us / avg_time_pctree_finger_multi_us;
    const double speedup_pctree_tri_multi = avg_time_hnsw_us / avg_time_pctree_tri_multi_us;
    const double speedup_mtree_multi = avg_time_hnsw_us / avg_time_mtree_multi_us;
    const double speedup_mtree_finger_multi = avg_time_hnsw_us / avg_time_mtree_finger_multi_us;
    const double speedup_mtree_tri_multi = avg_time_hnsw_us / avg_time_mtree_tri_multi_us;
    const double speedup_vpt_multi = avg_time_hnsw_us / avg_time_vpt_multi_us;
    const double speedup_vpt_finger_multi = avg_time_hnsw_us / avg_time_vpt_finger_multi_us;
    const double speedup_vpt_tri_multi = avg_time_hnsw_us / avg_time_vpt_tri_multi_us;

    std::cout << "\nSIFT1M RESULTS\n";
    std::cout << "N=" << N << ", dim=" << dim<< ", k=" << k << ", ef=" << ef_search<< ", queries=" << Q << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    auto print_result = [](const std::string& name, double recall,
                           double latency_us, double speedup) {
        std::cout << std::left << std::setw(28) << name
                  << " Recall@" << std::setw(2) << "k=" << std::setw(8) << recall
                  << " Avg latency=" << std::setw(10) << std::setprecision(3)
                  << latency_us << " us"
                  << " Speedup=" << std::setprecision(3) << speedup << "x\n";
    };



    print_result("Normal HNSW",avg_recall_hnsw,avg_time_hnsw_us,1.0);
    print_result("HNSW -> TRI",avg_recall_hnsw_tri,avg_time_hnsw_tri_us,speedup_hnsw_tri);
    print_result("HNSW + FINGER",avg_recall_hnsw_finger,avg_time_hnsw_finger_us,speedup_hnsw_finger);
    print_result("MTree -> HNSW",avg_recall_mtree,avg_time_mtree_us,speedup_mtree);
    print_result("MTree -> TRI",avg_recall_mtree_tri,avg_time_mtree_tri_us,speedup_mtree_tri);
    print_result("MTree -> FINGER",avg_recall_mtree_finger,avg_time_mtree_finger_us,speedup_mtree_finger);
    print_result("MTree -> HNSW Multi",avg_recall_mtree_multi,avg_time_mtree_multi_us,speedup_mtree_multi);
    print_result("MTree -> TRI Multi",avg_recall_mtree_tri_multi,avg_time_mtree_tri_multi_us,speedup_mtree_tri_multi);
    print_result("MTree -> FINGER Multi",avg_recall_mtree_finger_multi,avg_time_mtree_finger_multi_us,speedup_mtree_finger_multi);
    print_result("PCTree -> HNSW",avg_recall_pctree,avg_time_pctree_us,speedup_pctree);
    print_result("PCTree -> TRI",avg_recall_pctree_tri,avg_time_pctree_tri_us,speedup_pctree_tri);
    print_result("PCTree -> FINGER",avg_recall_pctree_finger,avg_time_pctree_finger_us,speedup_pctree_finger);
    print_result("PCTree -> HNSW Multi",avg_recall_pctree_multi,avg_time_pctree_multi_us,speedup_pctree_multi);
    print_result("PCTree -> TRI Multi",avg_recall_pctree_tri_multi,avg_time_pctree_tri_multi_us,speedup_pctree_tri_multi);
    print_result("PCTree -> FINGER Multi",avg_recall_pctree_finger_multi,avg_time_pctree_finger_multi_us,speedup_pctree_finger_multi);
    print_result("VPTree -> HNSW",avg_recall_vpt,avg_time_vpt_us,speedup_vpt);
    print_result("VPTree -> TRI",avg_recall_vpt_tri,avg_time_vpt_tri_us,speedup_vpt_tri);
    print_result("VPTree -> FINGER",avg_recall_vpt_finger,avg_time_vpt_finger_us,speedup_vpt_finger);
    print_result("VPTree -> HNSW Multi",avg_recall_vpt_multi,avg_time_vpt_multi_us,speedup_vpt_multi);
    print_result("VPTree -> TRI Multi",avg_recall_vpt_tri_multi,avg_time_vpt_tri_multi_us,speedup_vpt_tri_multi);
    print_result("VPTree -> FINGER Multi",avg_recall_vpt_finger_multi,avg_time_vpt_finger_multi_us,speedup_vpt_finger_multi);
    // print_result("VTree -> HNSW",avg_recall_vtree,avg_time_vtree_us,speedup_vtree);
    // print_result("VTree -> TRI",avg_recall_vtree_tri,avg_time_vtree_tri_us,speedup_vtree_tri);
    // print_result("VTree -> FINGER",avg_recall_vtree_finger,avg_time_vtree_finger_us,speedup_vtree_finger);
    print_result("KMeansTree -> HNSW",avg_recall_kmeans,avg_time_kmeans_us,speedup_kmeans);
    print_result("KMeansTree -> TRI",avg_recall_kmeans_tri,avg_time_kmeans_tri_us,speedup_kmeans_tri);
    print_result("KMeansTree -> FINGER",avg_recall_kmeans_finger,avg_time_kmeans_finger_us,speedup_kmeans_finger);
    print_result("KMeansTree -> HNSW Multi",avg_recall_kmeans_multi,avg_time_kmeans_multi_us,speedup_kmeans_multi);
    print_result("KMeansTree -> TRI Multi",avg_recall_kmeans_tri_multi,avg_time_kmeans_tri_multi_us,speedup_kmeans_tri_multi);
    print_result("KMeansTree -> FINGER Multi",avg_recall_kmeans_finger_multi,avg_time_kmeans_finger_multi_us,speedup_kmeans_finger_multi);


    // One summary row per search method. This file is overwritten on every run.
    const std::string csv_path = "hnsw_results_glove.csv";
    std::ofstream csv(csv_path);
    if (!csv) 
    {
        throw std::runtime_error("Cannot open " + csv_path + " for writing");
    }


    csv << "method,N,dim,k,ef_search,num_queries,M,ef_construction,"
    << "leaf_capacity,finger_rank,finger_warmup,"
    << "pctree_partitions,pctree_leaf_clusters,mtree_pivots,mtree_leaf_clusters,vpt_leaf_clusters,vt_pivots,"
    << "kmeans_clusters,kmeans_leaf_clusters,"
    << "hnsw_num_levels,pctree_height,mtree_height,vptree_height,"
    // << "vtree_height,
    << "kmeans_height,"
    << "hnsw_build_sec,pctree_build_sec,mtree_build_sec,"
    << "vptree_build_sec,"
    // <<"vtree_build_sec,"
    << "kmeans_build_sec,"
    << "finger_build_sec,tri_build_sec,recall_at_k,avg_latency_us,"
    << "speedup_vs_hnsw,recall_loss_vs_hnsw\n";



    auto write_row = [&](const std::string& method,double recall,double latency_us,double speedup)
    {
        csv << std::setprecision(10)
            << method << ','
            << N << ','
            << dim << ','
            << k << ','
            << ef_search << ','
            << Q << ','
            << M << ','
            << ef_construction << ','
            << leaf_capacity << ','
            << finger_rank << ','
            << finger_warmup << ','
            << pctree_partitions << ','
            << pctree_leaf_clusters << ','
            << mtree_pivots << ','
            << mtree_leaf_clusters << ','
            << vpt_leaf_clusters << ','
            // << vt_pivots << ','
            << kmeans_clusters << ','
            << kmeans_leaf_clusters<<','
            << hnsw_num_levels << ','
            << pctree_height << ','
            << mtree_height << ','
            << vptree_height << ','
            // << vtree_height << ','
            << kmeans_height << ','
            << build_sec << ','
            << pc_sec << ','
            << mtree_sec << ','
            << vpt_build_time << ','
            // << vtree_build_time << ','
            << kmeans_build_sec << ','
            << finger_build_sec << ','
            << tri_build_sec << ','
            << recall << ','
            << latency_us << ','
            << speedup << ','
            << (avg_recall_hnsw - recall)
            << '\n';
    };


    write_row("Normal HNSW", avg_recall_hnsw, avg_time_hnsw_us, 1.0);
    write_row("HNSW -> TRI", avg_recall_hnsw_tri, avg_time_hnsw_tri_us, speedup_hnsw_tri);
    write_row("HNSW + FINGER", avg_recall_hnsw_finger, avg_time_hnsw_finger_us, speedup_hnsw_finger);

    write_row("MTree -> HNSW", avg_recall_mtree, avg_time_mtree_us, speedup_mtree);
    write_row("MTree -> TRI", avg_recall_mtree_tri, avg_time_mtree_tri_us, speedup_mtree_tri);
    write_row("MTree -> FINGER", avg_recall_mtree_finger, avg_time_mtree_finger_us, speedup_mtree_finger);
    write_row("MTree -> HNSW Multi", avg_recall_mtree_multi, avg_time_mtree_multi_us, speedup_mtree_multi);
    write_row("MTree -> TRI Multi", avg_recall_mtree_tri_multi, avg_time_mtree_tri_multi_us, speedup_mtree_tri_multi);
    write_row("MTree -> FINGER Multi", avg_recall_mtree_finger_multi, avg_time_mtree_finger_multi_us, speedup_mtree_finger_multi);

    write_row("PCTree -> HNSW", avg_recall_pctree, avg_time_pctree_us, speedup_pctree);
    write_row("PCTree -> TRI", avg_recall_pctree_tri, avg_time_pctree_tri_us, speedup_pctree_tri);
    write_row("PCTree -> FINGER", avg_recall_pctree_finger, avg_time_pctree_finger_us, speedup_pctree_finger);
    write_row("PCTree -> HNSW Multi", avg_recall_pctree_multi, avg_time_pctree_multi_us, speedup_pctree_multi);
    write_row("PCTree -> TRI Multi", avg_recall_pctree_tri_multi, avg_time_pctree_tri_multi_us, speedup_pctree_tri_multi);
    write_row("PCTree -> FINGER Multi", avg_recall_pctree_finger_multi, avg_time_pctree_finger_multi_us, speedup_pctree_finger_multi);

    write_row("VPTree -> HNSW", avg_recall_vpt, avg_time_vpt_us, speedup_vpt);
    write_row("VPTree -> TRI", avg_recall_vpt_tri, avg_time_vpt_tri_us, speedup_vpt_tri);
    write_row("VPTree -> FINGER", avg_recall_vpt_finger, avg_time_vpt_finger_us, speedup_vpt_finger);
    write_row("VPTree -> HNSW Multi", avg_recall_vpt_multi, avg_time_vpt_multi_us, speedup_vpt_multi);
    write_row("VPTree -> TRI Multi", avg_recall_vpt_tri_multi, avg_time_vpt_tri_multi_us, speedup_vpt_tri_multi);
    write_row("VPTree -> FINGER Multi", avg_recall_vpt_finger_multi, avg_time_vpt_finger_multi_us, speedup_vpt_finger_multi);

    // write_row("VTree -> HNSW", avg_recall_vtree, avg_time_vtree_us, speedup_vtree);
    // write_row("VTree -> TRI", avg_recall_vtree_tri, avg_time_vtree_tri_us, speedup_vtree_tri);
    // write_row("VTree -> FINGER", avg_recall_vtree_finger, avg_time_vtree_finger_us, speedup_vtree_finger);

    write_row("KMeansTree -> HNSW",avg_recall_kmeans,avg_time_kmeans_us,speedup_kmeans);
    write_row("KMeansTree -> TRI",avg_recall_kmeans_tri,avg_time_kmeans_tri_us,speedup_kmeans_tri);
    write_row("KMeansTree -> FINGER",avg_recall_kmeans_finger,avg_time_kmeans_finger_us,speedup_kmeans_finger);
    write_row("KMeansTree -> HNSW Multi",avg_recall_kmeans_multi,avg_time_kmeans_multi_us,speedup_kmeans_multi);
    write_row("KMeansTree -> TRI Multi",avg_recall_kmeans_tri_multi,avg_time_kmeans_tri_multi_us,speedup_kmeans_tri_multi);
    write_row("KMeansTree -> FINGER Multi",avg_recall_kmeans_finger_multi,avg_time_kmeans_finger_multi_us,speedup_kmeans_finger_multi);

    csv.close();

    std::cout << "\nCSV written to: " << csv_path << "\n";
    return 0;
}