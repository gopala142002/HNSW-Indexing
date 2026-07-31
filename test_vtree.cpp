
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <cstdint>
#include "test_hnsw_shared.h"
#include "hnswlib/hnswlib.h"


std::vector<int> read_ivecs(const std::string& path, int& dim, int& n) 
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

    const std::streamsize record = 4 + static_cast<std::streamsize>(d) * 4;
    n = static_cast<int>(fsize / record);

    std::vector<int> data(static_cast<size_t>(n) * d);
    for (int i = 0;i<n;++i) 
    {
        int32_t dd;
        f.read(reinterpret_cast<char*>(&dd), 4);
        f.read(reinterpret_cast<char*>(&data[i * d]), 4 * d);
    }
    return data;
}

float recall_at_k(const int* gt, int gt_k, const std::vector<size_t>& returned, int k) 
{
    std::unordered_set<int> gt_set(gt, gt + std::min(gt_k, k));
    int hits = 0;
    for (size_t id : returned) 
    {
        if (gt_set.count(static_cast<int>(id))) 
            ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(std::min(gt_k, k));
}

int main(int argc, char** argv) {


    const std::string data_dir = (argc > 1) ? argv[1] : "./datasets/gist";
    const std::string base_path = data_dir + "/base.fvecs";
    const std::string query_path = data_dir + "/query.fvecs";
    const std::string gt_path = data_dir + "/groundtruth.ivecs";
  
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 50;
    const int k = 30;  
    const int num_queries = 10000;

    const int num_pivots = 5;  
    const int leaf_cap = 64;  
                                    
    std::cout << "Loading dataset base vectors from"<< base_path<<"\n";
    int base_dim, base_n;
    auto base=read_fvecs(base_path, base_dim, base_n);
    std::cout << "base: " << base_n << " x " << base_dim << "\n";

    std::cout << "Loading queries from " << query_path << "\n";
    int q_dim, q_n;
    auto queries = read_fvecs(query_path, q_dim, q_n);
    std::cout << "  queries: " << q_n << " x " << q_dim << "\n";

    if (base_dim != q_dim)
        throw std::runtime_error("Dimension mismatch between base and queries");

    std::cout << "Loading ground truth from " << gt_path << " ...\n";
    int gt_dim, gt_n;
    auto gt = read_ivecs(gt_path, gt_dim, gt_n);
    std::cout << "  gt: " << gt_n << " x " << gt_dim << "\n\n";

    const int Q = std::min(num_queries, q_n);
    const std::string cache_path = data_dir + "/sift_hnsw.bin";
    bool rebuilt = false;

    HnswCache cache = load_or_build_hnsw(data_dir, cache_path, const_cast<int&>(base_dim), const_cast<int&>(base_n), rebuilt);
    auto& hnsw = *cache.index;

    const int dim = base_dim;
    const int N   = base_n;

    hnsw.setEf(ef_search);


    std::cout << "\nBuilding VoronoiTree (numPivots=" << num_pivots
              << ", leafCapacity=" << leaf_cap << ")...\n";

    auto start = std::chrono::high_resolution_clock::now();

    hnsw.buildVoronoiTree();

    auto end   = std::chrono::high_resolution_clock::now();

    double total_time   = std::chrono::duration<double>(end - start).count();


    
    std::cout << "VoronoiTree built in " << std::setprecision(4) << total_time << " s.\n";

    std::cout << "VoronoiTree height: " << hnsw.getVTreeHeight() << "\n\n";

    std::cout << "Running " << Q << " queries (k=" << k << ", ef=" << ef_search << ")\n\n";

    double total_hnsw_recall = 0.0;
    double total_vtree_recall = 0.0;
    long total_query_time_hnsw = 0;
    long total_query_time_vtree = 0;

    for (int q = 0; q < Q; ++q) 
    {
        const float* query = queries.data() + static_cast<size_t>(q) * dim;
        const int*   gt_q  = gt.data()      + static_cast<size_t>(q) * gt_dim;


        auto t0 = std::chrono::high_resolution_clock::now();

        auto pq_hnsw = hnsw.searchKnn(query, static_cast<size_t>(k));

        auto t1 = std::chrono::high_resolution_clock::now();
        total_query_time_hnsw += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::vector<size_t> hnsw_ids;
        hnsw_ids.reserve(k);
        while (!pq_hnsw.empty()) {
            hnsw_ids.push_back(pq_hnsw.top().second);
            pq_hnsw.pop();
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        auto pq_vt = hnsw.searchKnnVTree(query, static_cast<size_t>(k));

        auto t3 = std::chrono::high_resolution_clock::now();
        total_query_time_vtree += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        std::vector<size_t> vtree_ids;
        vtree_ids.reserve(k);
        while (!pq_vt.empty()) {
            vtree_ids.push_back(pq_vt.top().second);
            pq_vt.pop();
        }

        total_hnsw_recall  += recall_at_k(gt_q, gt_dim, hnsw_ids,  k);
        total_vtree_recall += recall_at_k(gt_q, gt_dim, vtree_ids, k);
    }


    float avg_recall_hnsw  = static_cast<float>(total_hnsw_recall)  / Q;
    float avg_recall_vtree = static_cast<float>(total_vtree_recall) / Q;
    float avg_time_hnsw_us  = static_cast<float>(total_query_time_hnsw)  / Q;
    float avg_time_vtree_us = static_cast<float>(total_query_time_vtree) / Q;

        std::cout << std::fixed << std::setprecision(4);
        
        std::cout << "SIFT1M Results (N=" << N
              << ", dim=" << dim
              << ", k=" << k
              << ", ef=" << ef_search << ")\n\n";

        std::cout << "HNSW (upper layers): Recall@" << k << " = " << avg_recall_hnsw
            << ", Avg latency (us) = " << std::fixed << std::setprecision(1) << avg_time_hnsw_us << "\n";
        std::cout << "VTree -> level-0:    Recall@" << k << " = " << std::setprecision(4) << avg_recall_vtree
            << ", Avg latency (us) = " << std::fixed << std::setprecision(1) << avg_time_vtree_us << "\n";

            std::cout << "\n";
        std::cout << "Speedup (HNSW / PCTree):     " << std::fixed << std::setprecision(2)
                << (avg_time_hnsw_us / avg_time_vtree_us) << "x\n";
        std::cout << "Recall loss (HNSW - PCTree): " << std::fixed << std::setprecision(4)
                << (avg_recall_hnsw - avg_recall_vtree) << "\n";

    return 0;
}