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
    for(int i = 0; i < n; ++i) 
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
    const std::string data_dir =(argc > 1)? argv[1]: "./datasets/gist";
    const std::string base_path =data_dir + "/base.fvecs";
    const std::string query_path =data_dir + "/query.fvecs";
    const std::string gt_path =data_dir + "/groundtruth.ivecs";




    // Parameters
    const int M = 16;
    const int ef_construction = 200;



    // Search-effort values to benchmark. The graph and auxiliary trees are built once.
    const std::vector<int> ef_search_values = {200};
    const std::vector<int> k_values = {30};
    const int num_queries = 10000;


    const int leaf_capacity = 64;
    const int finger_rank = 16;
    const int finger_warmup = 16;


    const int pctree_partitions = 4;
    const int mtree_pivots=4;
    const int kmeans_clusters = 4;
    const int kmeans_leaf_clusters = 1;
    const int mtree_leaf_clusters = 1;
    const int vpt_leaf_clusters = 1;
    const int pctree_leaf_clusters = 1;



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

    // VTree skipped: not performing well on higher dimensions and
    // buildVoronoiTree/searchKnnVTree* are currently disabled in HierarchicalNSW.
    
    
    
    // Building KMeansTree
    std::cout << "\nBuilding KMeansTree "<< "(K=" << kmeans_clusters<< ", leafCapacity=" << leaf_capacity << ", leafClusters=" << kmeans_leaf_clusters << ")...\n";
    auto t_kmeans_start = std::chrono::high_resolution_clock::now();
    hnsw.buildKMeansTree(kmeans_clusters, leaf_capacity, kmeans_leaf_clusters);
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
    double tri_build_sec = std::chrono::duration<double>(t_tri_end - t_tri_start).count();
    std::cout << "TRI structures built in " << std::fixed << std::setprecision(2) << tri_build_sec << " s.\n\n";
    profilingStats profiler;

    
    for (int leafClusters = 4; leafClusters <= 4; ++leafClusters)
    {
        std::cout << "\nRecomputing leaf clusters = " << leafClusters << "\n";
        hnsw.recomputeAllLeafClusters(leafClusters);
        const std::string csv_path ="temp_hnsw_results_gist_leaf_" + std::to_string(leafClusters) + ".csv";
        bool first_csv_write = true;
        for (int k : k_values) {
            for (int ef_search : ef_search_values) {
                hnsw.setEf(ef_search);
                profiler.reset(hnsw_num_levels);
                std::cout << "Running " << Q << " queries (k=" << k << ", ef=" << ef_search << ")...\n";
                double total_recall_hnsw = 0.0;
                double total_recall_hnsw_finger = 0.0;
                double total_recall_pctree = 0.0;
                double total_recall_pctree_finger = 0.0;
                double total_recall_pctree_tri = 0.0;
                double total_recall_mtree_tri = 0.0;
                double total_recall_vpt_tri = 0.0;
                double total_recall_hnsw_tri = 0.0;
                double total_recall_mtree = 0.0;
                double total_recall_mtree_finger = 0.0;
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
                double time_pctree_ns = 0.0;
                double time_pctree_finger_ns = 0.0;
                double time_pctree_tri_ns = 0.0;
                double time_mtree_tri_ns = 0.0;
                double time_vpt_tri_ns = 0.0;
                double time_hnsw_tri_ns = 0.0;
                double time_mtree_ns = 0.0;
                double time_mtree_finger_ns = 0.0;
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


                std::cout << "Running HNSW TRI for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnTri(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_hnsw_tri_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_hnsw_tri += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running HNSW FINGER for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnFinger(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_hnsw_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_hnsw_finger += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running HNSW NORMAL for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnn(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_hnsw_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_hnsw += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree TRI for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTreeTri(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_tri_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree_tri += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree FINGER for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTreeFinger(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree_finger += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree NORMAL for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTree(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree TRI Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTreeTriMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_tri_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree_tri_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree FINGER Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTreeFingerMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_finger_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree_finger_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running MTree NORMAL Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnMTreeMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_mtree_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_mtree_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree TRI for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTreeTri(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_tri_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree_tri += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree FINGER for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTreeFinger(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree_finger += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree NORMAL for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTree(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree TRI Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTreeTriMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_tri_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree_tri_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree FINGER Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTreeFingerMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_finger_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree_finger_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running PCTree NORMAL Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnPCTreeMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_pctree_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_pctree_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree TRI for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTreeTri(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_tri_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt_tri += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree FINGER for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTreeFinger(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt_finger += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree NORMAL for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTree(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree TRI Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTreeTriMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_tri_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt_tri_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree FINGER Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTreeFingerMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_finger_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt_finger_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running VPTree NORMAL Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnVPTreeMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_vpt_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_vpt_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree TRI for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTreeTri(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_tri_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans_tri += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree FINGER for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTreeFinger(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans_finger += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree NORMAL for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTree(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree TRI Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTreeTriMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_tri_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans_tri_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree FINGER Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTreeFingerMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_finger_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans_finger_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }

                std::cout << "Running KMeansTree NORMAL Multi for " << Q << " queries...\n";
                for (int q = 0; q < Q; ++q) {
                    const float* query = queries.data() + static_cast<size_t>(q) * dim;
                    const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;
                    auto start = std::chrono::high_resolution_clock::now();
                    auto pq = hnsw.searchKnnKMeansTreeMulti(query, static_cast<size_t>(k), nullptr, &profiler);
                    auto end = std::chrono::high_resolution_clock::now();
                    time_kmeans_multi_ns += std::chrono::duration<double, std::nano>(end - start).count();
                    auto ids = extract_ids(pq, k);
                    total_recall_kmeans_multi += recall_at_k(gt_q, gt_dim, ids, k);
                }



                const double avg_recall_hnsw = total_recall_hnsw / Q;
                const double avg_recall_hnsw_finger =total_recall_hnsw_finger / Q;
                const double avg_recall_pctree = total_recall_pctree / Q;
                const double avg_recall_pctree_finger = total_recall_pctree_finger / Q;
                const double avg_recall_pctree_tri = total_recall_pctree_tri / Q;
                const double avg_recall_hnsw_tri = total_recall_hnsw_tri / Q;
                const double avg_recall_mtree_tri = total_recall_mtree_tri / Q;
                const double avg_recall_vpt_tri = total_recall_vpt_tri / Q;
                const double avg_recall_mtree = total_recall_mtree / Q;
                const double avg_recall_mtree_finger = total_recall_mtree_finger / Q;
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

                // Per-method latency in NANOSECONDS (no /1000 conversion).

                const double avg_time_hnsw_ns = time_hnsw_ns / Q;
                const double avg_time_hnsw_finger_ns = time_hnsw_finger_ns / Q;
                const double avg_time_pctree_ns = time_pctree_ns / Q;
                const double avg_time_pctree_finger_ns = time_pctree_finger_ns / Q;
                const double avg_time_pctree_tri_ns = time_pctree_tri_ns / Q;
                const double avg_time_hnsw_tri_ns = time_hnsw_tri_ns / Q;
                const double avg_time_mtree_tri_ns = time_mtree_tri_ns / Q;
                const double avg_time_vpt_tri_ns = time_vpt_tri_ns / Q;
                const double avg_time_mtree_ns = time_mtree_ns / Q;
                const double avg_time_mtree_finger_ns = time_mtree_finger_ns / Q;
                const double avg_time_vpt_ns = time_vpt_ns / Q;
                const double avg_time_vpt_finger_ns = time_vpt_finger_ns / Q;
                const double avg_time_kmeans_ns = time_kmeans_ns / Q;
                const double avg_time_kmeans_finger_ns = time_kmeans_finger_ns / Q;
                const double avg_time_kmeans_tri_ns = time_kmeans_tri_ns / Q;
                const double avg_time_kmeans_multi_ns = time_kmeans_multi_ns / Q;
                const double avg_time_kmeans_finger_multi_ns = time_kmeans_finger_multi_ns / Q;
                const double avg_time_kmeans_tri_multi_ns = time_kmeans_tri_multi_ns / Q;
                const double avg_time_pctree_multi_ns = time_pctree_multi_ns / Q;
                const double avg_time_pctree_finger_multi_ns = time_pctree_finger_multi_ns / Q;
                const double avg_time_pctree_tri_multi_ns = time_pctree_tri_multi_ns / Q;
                const double avg_time_mtree_multi_ns = time_mtree_multi_ns / Q;
                const double avg_time_mtree_finger_multi_ns = time_mtree_finger_multi_ns / Q;
                const double avg_time_mtree_tri_multi_ns = time_mtree_tri_multi_ns / Q;
                const double avg_time_vpt_multi_ns = time_vpt_multi_ns / Q;
                const double avg_time_vpt_finger_multi_ns = time_vpt_finger_multi_ns / Q;
                const double avg_time_vpt_tri_multi_ns = time_vpt_tri_multi_ns / Q;

                // Microsecond versions kept only for the human-readable console printout.

                const double avg_time_hnsw_us = avg_time_hnsw_ns / 1000.0;
                const double avg_time_hnsw_finger_us = avg_time_hnsw_finger_ns / 1000.0;
                const double avg_time_pctree_us = avg_time_pctree_ns / 1000.0;
                const double avg_time_pctree_finger_us = avg_time_pctree_finger_ns / 1000.0;
                const double avg_time_pctree_tri_us = avg_time_pctree_tri_ns / 1000.0;
                const double avg_time_hnsw_tri_us = avg_time_hnsw_tri_ns / 1000.0;
                const double avg_time_mtree_tri_us = avg_time_mtree_tri_ns / 1000.0;
                const double avg_time_vpt_tri_us = avg_time_vpt_tri_ns / 1000.0;
                const double avg_time_mtree_us = avg_time_mtree_ns / 1000.0;
                const double avg_time_mtree_finger_us = avg_time_mtree_finger_ns / 1000.0;
                const double avg_time_vpt_us = avg_time_vpt_ns / 1000.0;
                const double avg_time_vpt_finger_us = avg_time_vpt_finger_ns / 1000.0;
                const double avg_time_kmeans_us = avg_time_kmeans_ns / 1000.0;
                const double avg_time_kmeans_finger_us = avg_time_kmeans_finger_ns / 1000.0;
                const double avg_time_kmeans_tri_us = avg_time_kmeans_tri_ns / 1000.0;
                const double avg_time_kmeans_multi_us = avg_time_kmeans_multi_ns / 1000.0;
                const double avg_time_kmeans_finger_multi_us = avg_time_kmeans_finger_multi_ns / 1000.0;
                const double avg_time_kmeans_tri_multi_us = avg_time_kmeans_tri_multi_ns / 1000.0;
                const double avg_time_pctree_multi_us = avg_time_pctree_multi_ns / 1000.0;
                const double avg_time_pctree_finger_multi_us = avg_time_pctree_finger_multi_ns / 1000.0;
                const double avg_time_pctree_tri_multi_us = avg_time_pctree_tri_multi_ns / 1000.0;
                const double avg_time_mtree_multi_us = avg_time_mtree_multi_ns / 1000.0;
                const double avg_time_mtree_finger_multi_us = avg_time_mtree_finger_multi_ns / 1000.0;
                const double avg_time_mtree_tri_multi_us = avg_time_mtree_tri_multi_ns / 1000.0;
                const double avg_time_vpt_multi_us = avg_time_vpt_multi_ns / 1000.0;
                const double avg_time_vpt_finger_multi_us = avg_time_vpt_finger_multi_ns / 1000.0;
                const double avg_time_vpt_tri_multi_us = avg_time_vpt_tri_multi_ns / 1000.0;
                const double speedup_pctree = avg_time_hnsw_ns / avg_time_pctree_ns;
                const double speedup_hnsw_finger = avg_time_hnsw_ns / avg_time_hnsw_finger_ns;
                const double speedup_pctree_finger = avg_time_hnsw_ns / avg_time_pctree_finger_ns;
                const double speedup_pctree_tri = avg_time_hnsw_ns / avg_time_pctree_tri_ns;
                const double speedup_hnsw_tri = avg_time_hnsw_ns / avg_time_hnsw_tri_ns;
                const double speedup_mtree_tri = avg_time_hnsw_ns / avg_time_mtree_tri_ns;
                const double speedup_vpt_tri = avg_time_hnsw_ns / avg_time_vpt_tri_ns;
                const double speedup_mtree = avg_time_hnsw_ns / avg_time_mtree_ns;
                const double speedup_mtree_finger = avg_time_hnsw_ns / avg_time_mtree_finger_ns;
                const double speedup_vpt = avg_time_hnsw_ns / avg_time_vpt_ns;
                const double speedup_vpt_finger = avg_time_hnsw_ns / avg_time_vpt_finger_ns;
                const double speedup_kmeans = avg_time_hnsw_ns / avg_time_kmeans_ns;
                const double speedup_kmeans_finger = avg_time_hnsw_ns / avg_time_kmeans_finger_ns;
                const double speedup_kmeans_tri = avg_time_hnsw_ns / avg_time_kmeans_tri_ns;
                const double speedup_kmeans_multi = avg_time_hnsw_ns / avg_time_kmeans_multi_ns;
                const double speedup_kmeans_finger_multi = avg_time_hnsw_ns / avg_time_kmeans_finger_multi_ns;
                const double speedup_kmeans_tri_multi = avg_time_hnsw_ns / avg_time_kmeans_tri_multi_ns;
                const double speedup_pctree_multi = avg_time_hnsw_ns / avg_time_pctree_multi_ns;
                const double speedup_pctree_finger_multi = avg_time_hnsw_ns / avg_time_pctree_finger_multi_ns;
                const double speedup_pctree_tri_multi = avg_time_hnsw_ns / avg_time_pctree_tri_multi_ns;
                const double speedup_mtree_multi = avg_time_hnsw_ns / avg_time_mtree_multi_ns;
                const double speedup_mtree_finger_multi = avg_time_hnsw_ns / avg_time_mtree_finger_multi_ns;
                const double speedup_mtree_tri_multi = avg_time_hnsw_ns / avg_time_mtree_tri_multi_ns;
                const double speedup_vpt_multi = avg_time_hnsw_ns / avg_time_vpt_multi_ns;
                const double speedup_vpt_finger_multi = avg_time_hnsw_ns / avg_time_vpt_finger_multi_ns;
                const double speedup_vpt_tri_multi = avg_time_hnsw_ns / avg_time_vpt_tri_multi_ns;

                // Averaged profiling values for this ef_search value, stored in NANOSECONDS.

                std::vector<double> avg_hnsw_level_time_ns(profiler.hnswLevelTimeNs.size(), 0.0);
                for (size_t lvl = 0; lvl < profiler.hnswLevelTimeNs.size(); ++lvl)
                {
                    avg_hnsw_level_time_ns[lvl] = profiler.hnswLevelTimeNs[lvl] / Q;
                }

                const double avg_hnsw_tri_level0_ns    = profiler.hnswTriLevel0TimeNs    / Q;
                const double avg_hnsw_finger_level0_ns = profiler.hnswFingerLevel0TimeNs / Q;
                const double avg_pctree_lookup_ns      = profiler.pctreeTimeNs          / Q;
                const double avg_mtree_lookup_ns       = profiler.mtreeTimeNs           / Q;
                const double avg_vptree_lookup_ns      = profiler.vptreeTimeNs          / Q;
                const double avg_kmeans_tree_lookup_ns = profiler.kmeansTreeTimeNs      / Q;

                // Per-tree, per-scheme HNSW level-0 breakdown (plain / TRI / FINGER)

                const double avg_hnswLevel0_PCTree_ns     = profiler.hnswLevel0_PCTree    / Q;
                const double avg_hnswLevel0_MTree_ns      = profiler.hnswLevel0_MTree     / Q;
                const double avg_hnswLevel0_VPTree_ns     = profiler.hnswLevel0_VPTree    / Q;
                const double avg_hnswLevel0_KMeanTree_ns  = profiler.hnswLevel0_KMeanTree / Q;
                const double avg_hnswTriLevel0_PCTree_ns    = profiler.hnswTriLevel0_PCTree    / Q;
                const double avg_hnswTriLevel0_MTree_ns     = profiler.hnswTriLevel0_MTree     / Q;
                const double avg_hnswTriLevel0_VPTree_ns    = profiler.hnswTriLevel0_VPTree    / Q;
                const double avg_hnswTriLevel0_KMeanTree_ns = profiler.hnswTriLevel0_KMeanTree / Q;
                const double avg_hnswFingerLevel0_PCTree_ns    = profiler.hnswFingerLevel0_PCTree    / Q;
                const double avg_hnswFingerLevel0_MTree_ns     = profiler.hnswFingerLevel0_MTree     / Q;
                const double avg_hnswFingerLevel0_VPTree_ns    = profiler.hnswFingerLevel0_VPTree    / Q;
                const double avg_hnswFingerLevel0_KMeanTree_ns = profiler.hnswFingerLevel0_KMeanTree / Q;

                // Multi-entry-point variants, per tree / scheme

                const double avg_hnswLevel0PCTreeMulti_ns       = profiler.hnswLevel0PCTreeMulti       / Q;
                const double avg_hnswFingerLevel0PCTreeMulti_ns = profiler.hnswFingerLevel0PCTreeMulti / Q;
                const double avg_hnswTriLevel0PCTreeMulti_ns    = profiler.hnswTriLevel0PCTreeMulti    / Q;
                const double avg_hnswLevel0MTreeMulti_ns       = profiler.hnswLevel0MTreeMulti       / Q;
                const double avg_hnswFingerLevel0MTreeMulti_ns = profiler.hnswFingerLevel0MTreeMulti / Q;
                const double avg_hnswTriLevel0MTreeMulti_ns    = profiler.hnswTriLevel0MTreeMulti    / Q;
                const double avg_hnswLevel0VPTreeMulti_ns       = profiler.hnswLevel0VPTreeMulti       / Q;
                const double avg_hnswFingerLevel0VPTreeMulti_ns = profiler.hnswFingerLevel0VPTreeMulti / Q;
                const double avg_hnswTriLevel0VPTreeMulti_ns    = profiler.hnswTriLevel0VPTreeMulti    / Q;
                const double avg_hnswLevel0KMeansTreeMulti_ns       = profiler.hnswLevel0KMeansTreeMulti       / Q;
                const double avg_hnswFingerLevel0KMeansTreeMulti_ns = profiler.hnswFingerLevel0KMeansTreeMulti / Q;
                const double avg_hnswTriLevel0KMeansTreeMulti_ns    = profiler.hnswTriLevel0KMeansTreeMulti    / Q;

                std::cout << "\nGlOVE RESULTS\n";
                std::cout << "N=" << N << ", dim=" << dim<< ", k=" << k << ", ef=" << ef_search<< ", queries=" << Q << "\n\n";
                std::cout << std::fixed << std::setprecision(4);
                auto print_result = [](const std::string& name, double recall,double latency_us, double speedup) 
                {
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
                print_result("KMeansTree -> HNSW",avg_recall_kmeans,avg_time_kmeans_us,speedup_kmeans);
                print_result("KMeansTree -> TRI",avg_recall_kmeans_tri,avg_time_kmeans_tri_us,speedup_kmeans_tri);
                print_result("KMeansTree -> FINGER",avg_recall_kmeans_finger,avg_time_kmeans_finger_us,speedup_kmeans_finger);
                print_result("KMeansTree -> HNSW Multi",avg_recall_kmeans_multi,avg_time_kmeans_multi_us,speedup_kmeans_multi);
                print_result("KMeansTree -> TRI Multi",avg_recall_kmeans_tri_multi,avg_time_kmeans_tri_multi_us,speedup_kmeans_tri_multi);
                print_result("KMeansTree -> FINGER Multi",avg_recall_kmeans_finger_multi,avg_time_kmeans_finger_multi_us,speedup_kmeans_finger_multi);

                // Append one result block for this ef_search value. Single CSV, all timing in NANOSECONDS.

                std::ofstream csv(csv_path,first_csv_write ? std::ios::out : std::ios::app);
                if (!csv) 
                {
                    throw std::runtime_error("Cannot open " + csv_path + " for writing");
                }

                if (first_csv_write)
                {
                    csv << "method,N,dim,k,ef_search,num_queries,M,ef_construction,"
                        << "leaf_capacity,finger_rank,finger_warmup,"
                        << "pctree_partitions,pctree_leaf_clusters,mtree_pivots,mtree_leaf_clusters,"
                        << "vpt_leaf_clusters,kmeans_clusters,kmeans_leaf_clusters,"
                        << "hnsw_num_levels,pctree_height,mtree_height,vptree_height,"
                        << "kmeans_height,"
                        << "hnsw_build_sec,pctree_build_sec,mtree_build_sec,"
                        << "vptree_build_sec,kmeans_build_sec,"
                        << "finger_build_sec,tri_build_sec,recall_at_k,avg_latency_ns,"
                        << "speedup_vs_hnsw,recall_loss_vs_hnsw";
                    for (int lvl = 0; lvl < hnsw_num_levels; ++lvl)
                    {
                        csv << ",avg_hnsw_level_" << lvl << "_time_ns";
                    }

                    csv << ",avg_hnsw_tri_level0_time_ns"
                        << ",avg_hnsw_finger_level0_time_ns"
                        << ",avg_pctree_lookup_time_ns"
                        << ",avg_mtree_lookup_time_ns"
                        << ",avg_vptree_lookup_time_ns"
                        << ",avg_kmeans_tree_lookup_time_ns"
                        << ",avg_hnswLevel0_PCTree_ns"
                        << ",avg_hnswLevel0_MTree_ns"
                        << ",avg_hnswLevel0_VPTree_ns"
                        << ",avg_hnswLevel0_KMeanTree_ns"
                        << ",avg_hnswTriLevel0_PCTree_ns"
                        << ",avg_hnswTriLevel0_MTree_ns"
                        << ",avg_hnswTriLevel0_VPTree_ns"
                        << ",avg_hnswTriLevel0_KMeanTree_ns"
                        << ",avg_hnswFingerLevel0_PCTree_ns"
                        << ",avg_hnswFingerLevel0_MTree_ns"
                        << ",avg_hnswFingerLevel0_VPTree_ns"
                        << ",avg_hnswFingerLevel0_KMeanTree_ns"
                        << ",avg_hnswLevel0PCTreeMulti_ns"
                        << ",avg_hnswFingerLevel0PCTreeMulti_ns"
                        << ",avg_hnswTriLevel0PCTreeMulti_ns"
                        << ",avg_hnswLevel0MTreeMulti_ns"
                        << ",avg_hnswFingerLevel0MTreeMulti_ns"
                        << ",avg_hnswTriLevel0MTreeMulti_ns"
                        << ",avg_hnswLevel0VPTreeMulti_ns"
                        << ",avg_hnswFingerLevel0VPTreeMulti_ns"
                        << ",avg_hnswTriLevel0VPTreeMulti_ns"
                        << ",avg_hnswLevel0KMeansTreeMulti_ns"
                        << ",avg_hnswFingerLevel0KMeansTreeMulti_ns"
                        << ",avg_hnswTriLevel0KMeansTreeMulti_ns\n";
                        first_csv_write = false;
                }

                auto write_row = [&](const std::string& method,double recall,double latency_ns,double speedup)
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
                    << leafClusters << ','
                    << mtree_pivots << ','
                    << leafClusters << ','
                    << leafClusters << ','
                    << kmeans_clusters << ','
                    << leafClusters << ','
                    << hnsw_num_levels << ','
                    << pctree_height << ','
                    << mtree_height << ','
                    << vptree_height << ','
                    << kmeans_height << ','
                    << build_sec << ','
                    << pc_sec << ','
                    << mtree_sec << ','
                    << vpt_build_time << ','
                    << kmeans_build_sec << ','
                    << finger_build_sec << ','
                    << tri_build_sec << ','
                    << recall << ','
                    << latency_ns << ','
                    << speedup << ','
                    << (avg_recall_hnsw - recall);
                for (int lvl = 0; lvl < hnsw_num_levels; ++lvl)
                {
                    csv << ',' << avg_hnsw_level_time_ns[lvl];
                }
                csv << ',' << avg_hnsw_tri_level0_ns
                    << ',' << avg_hnsw_finger_level0_ns
                    << ',' << avg_pctree_lookup_ns
                    << ',' << avg_mtree_lookup_ns
                    << ',' << avg_vptree_lookup_ns
                    << ',' << avg_kmeans_tree_lookup_ns
                    << ',' << avg_hnswLevel0_PCTree_ns
                    << ',' << avg_hnswLevel0_MTree_ns
                    << ',' << avg_hnswLevel0_VPTree_ns
                    << ',' << avg_hnswLevel0_KMeanTree_ns
                    << ',' << avg_hnswTriLevel0_PCTree_ns
                    << ',' << avg_hnswTriLevel0_MTree_ns
                    << ',' << avg_hnswTriLevel0_VPTree_ns
                    << ',' << avg_hnswTriLevel0_KMeanTree_ns
                    << ',' << avg_hnswFingerLevel0_PCTree_ns
                    << ',' << avg_hnswFingerLevel0_MTree_ns
                    << ',' << avg_hnswFingerLevel0_VPTree_ns
                    << ',' << avg_hnswFingerLevel0_KMeanTree_ns
                    << ',' << avg_hnswLevel0PCTreeMulti_ns
                    << ',' << avg_hnswFingerLevel0PCTreeMulti_ns
                    << ',' << avg_hnswTriLevel0PCTreeMulti_ns
                    << ',' << avg_hnswLevel0MTreeMulti_ns
                    << ',' << avg_hnswFingerLevel0MTreeMulti_ns
                    << ',' << avg_hnswTriLevel0MTreeMulti_ns
                    << ',' << avg_hnswLevel0VPTreeMulti_ns
                    << ',' << avg_hnswFingerLevel0VPTreeMulti_ns
                    << ',' << avg_hnswTriLevel0VPTreeMulti_ns
                    << ',' << avg_hnswLevel0KMeansTreeMulti_ns
                    << ',' << avg_hnswFingerLevel0KMeansTreeMulti_ns
                    << ',' << avg_hnswTriLevel0KMeansTreeMulti_ns
                    << '\n';
            };

                write_row("Normal HNSW", avg_recall_hnsw, avg_time_hnsw_ns, 1.0);
                write_row("HNSW -> TRI", avg_recall_hnsw_tri, avg_time_hnsw_tri_ns, speedup_hnsw_tri);
                write_row("HNSW + FINGER", avg_recall_hnsw_finger, avg_time_hnsw_finger_ns, speedup_hnsw_finger);
                write_row("MTree -> HNSW", avg_recall_mtree, avg_time_mtree_ns, speedup_mtree);
                write_row("MTree -> TRI", avg_recall_mtree_tri, avg_time_mtree_tri_ns, speedup_mtree_tri);
                write_row("MTree -> FINGER", avg_recall_mtree_finger, avg_time_mtree_finger_ns, speedup_mtree_finger);
                write_row("MTree -> HNSW Multi", avg_recall_mtree_multi, avg_time_mtree_multi_ns, speedup_mtree_multi);
                write_row("MTree -> TRI Multi", avg_recall_mtree_tri_multi, avg_time_mtree_tri_multi_ns, speedup_mtree_tri_multi);
                write_row("MTree -> FINGER Multi", avg_recall_mtree_finger_multi, avg_time_mtree_finger_multi_ns, speedup_mtree_finger_multi);
                write_row("PCTree -> HNSW", avg_recall_pctree, avg_time_pctree_ns, speedup_pctree);
                write_row("PCTree -> TRI", avg_recall_pctree_tri, avg_time_pctree_tri_ns, speedup_pctree_tri);
                write_row("PCTree -> FINGER", avg_recall_pctree_finger, avg_time_pctree_finger_ns, speedup_pctree_finger);
                write_row("PCTree -> HNSW Multi", avg_recall_pctree_multi, avg_time_pctree_multi_ns, speedup_pctree_multi);
                write_row("PCTree -> TRI Multi", avg_recall_pctree_tri_multi, avg_time_pctree_tri_multi_ns, speedup_pctree_tri_multi);
                write_row("PCTree -> FINGER Multi", avg_recall_pctree_finger_multi, avg_time_pctree_finger_multi_ns, speedup_pctree_finger_multi);
                write_row("VPTree -> HNSW", avg_recall_vpt, avg_time_vpt_ns, speedup_vpt);
                write_row("VPTree -> TRI", avg_recall_vpt_tri, avg_time_vpt_tri_ns, speedup_vpt_tri);
                write_row("VPTree -> FINGER", avg_recall_vpt_finger, avg_time_vpt_finger_ns, speedup_vpt_finger);
                write_row("VPTree -> HNSW Multi", avg_recall_vpt_multi, avg_time_vpt_multi_ns, speedup_vpt_multi);
                write_row("VPTree -> TRI Multi", avg_recall_vpt_tri_multi, avg_time_vpt_tri_multi_ns, speedup_vpt_tri_multi);
                write_row("VPTree -> FINGER Multi", avg_recall_vpt_finger_multi, avg_time_vpt_finger_multi_ns, speedup_vpt_finger_multi);
                write_row("KMeansTree -> HNSW",avg_recall_kmeans,avg_time_kmeans_ns,speedup_kmeans);
                write_row("KMeansTree -> TRI",avg_recall_kmeans_tri,avg_time_kmeans_tri_ns,speedup_kmeans_tri);
                write_row("KMeansTree -> FINGER",avg_recall_kmeans_finger,avg_time_kmeans_finger_ns,speedup_kmeans_finger);
                write_row("KMeansTree -> HNSW Multi",avg_recall_kmeans_multi,avg_time_kmeans_multi_ns,speedup_kmeans_multi);
                write_row("KMeansTree -> TRI Multi",avg_recall_kmeans_tri_multi,avg_time_kmeans_tri_multi_ns,speedup_kmeans_tri_multi);
                write_row("KMeansTree -> FINGER Multi",avg_recall_kmeans_finger_multi,avg_time_kmeans_finger_multi_ns,speedup_kmeans_finger_multi);
                csv.close();
            std::cout << "EF=" << ef_search << " results appended to " << csv_path << "\n";
            }
        }
        std::cout << "Completed leafClusters=" << leafClusters << ". CSV written to: " << csv_path << "\n";
    }
    std::cout << "\nExperiment completed for leafClusters=4, k=30, ef=200.\n";
    return 0;

}