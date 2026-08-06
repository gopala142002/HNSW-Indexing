#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnswlib/hnswlib.h"



std::vector<float> read_fvecs(const std::string& path, int& dim, int& n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    int32_t d;
    f.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    f.seekg(0, std::ios::end);
    std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    const std::streamsize record = 4 + static_cast<std::streamsize>(d) * 4;
    n = static_cast<int>(fsize / record);

    std::vector<float> data(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; ++i) {
        int32_t dd;
        f.read(reinterpret_cast<char*>(&dd), 4);
        f.read(reinterpret_cast<char*>(&data[i * d]), 4 * d);
    }
    return data;
}

std::vector<int> read_ivecs(const std::string& path, int& dim, int& n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    int32_t d;
    f.read(reinterpret_cast<char*>(&d), 4);
    dim = d;

    f.seekg(0, std::ios::end);
    std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    const std::streamsize record = 4 + static_cast<std::streamsize>(d) * 4;
    n = static_cast<int>(fsize / record);

    std::vector<int> data(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; ++i) {
        int32_t dd;
        f.read(reinterpret_cast<char*>(&dd), 4);
        f.read(reinterpret_cast<char*>(&data[i * d]), 4 * d);
    }
    return data;
}

float recall_at_k(const int* gt, int gt_k, const std::vector<size_t>& returned, int k) {
    std::unordered_set<int> gt_set(gt, gt + std::min(gt_k, k));
    int hits = 0;
    for (size_t id : returned) {
        if (gt_set.count(static_cast<int>(id))) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(std::min(gt_k, k));
}


int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "./datasets/gist";
    const std::string base_path = data_dir + "/base.fvecs";
    const std::string query_path = data_dir + "/query.fvecs";
    const std::string gt_path = data_dir + "/groundtruth.ivecs";

    // Parse command-line parameters
    // Usage: test_mtree <dataset_path> [M] [ef_construction] [ef_search]
    int M = 16;
    int ef_construction = 200;
    int ef_search = 50;
    
    if (argc > 2) M = std::atoi(argv[2]);
    if (argc > 3) ef_construction = std::atoi(argv[3]);
    if (argc > 4) ef_search = std::atoi(argv[4]);
    
    const int k = 30;
    const int num_queries = 10000;

    std::cout << "Loading SIFT1M base vectors from " << base_path << " ...\n";
    int base_dim, base_n;
    auto base = read_fvecs(base_path, base_dim, base_n);
    std::cout << "  base: " << base_n << " x " << base_dim << "\n";

    std::cout << "Loading queries from " << query_path << " ...\n";
    int q_dim, q_n;
    auto queries = read_fvecs(query_path, q_dim, q_n);
    std::cout << "  queries: " << q_n << " x " << q_dim << "\n";

    // if (base_dim != q_dim) 
    // {
    //     throw std::runtime_error("Dimension mismatch between base and queries");
    // }

    std::cout << "Loading ground truth from " << gt_path << " ...\n";
    int gt_dim, gt_n;
    auto gt = read_ivecs(gt_path, gt_dim, gt_n);
    std::cout << " gt: " << gt_n << " x " << gt_dim << "\n\n";

    const int dim = base_dim;
    const int N = base_n;
    const int Q = std::min(num_queries, q_n);

    const std::string hnsw_bin = data_dir + "/sift_hnsw.bin";
    std::cout << "Building/loading HNSW index over " << N << " vectors (M=" << M
              << ", ef_construction=" << ef_construction << ")...\n";

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, ef_construction);

    auto t_build_start = std::chrono::high_resolution_clock::now();
    std::ifstream index_file(hnsw_bin, std::ios::binary);
    if (index_file.good()) 
    {
        hnsw.loadIndex(hnsw_bin, &space, N);
        std::cout << "Loaded existing HNSW index from " << hnsw_bin << "\n";
    } 
    else 
    {
        for (int i = 0; i < N; ++i) 
        {
            hnsw.addPoint(base.data() + static_cast<size_t>(i) * dim, static_cast<size_t>(i));
            if ((i + 1) % 100000 == 0) 
            {
                std::cout << "  indexed " << (i + 1) << " / " << N << "\n";
            }
        }
    }
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << "HNSW ready in " << std::fixed << std::setprecision(1)
              << build_sec << " s.  Max level = " << hnsw.maxlevel_ << "\n";

    hnsw.setEf(ef_search);

    std::cout << "\nBuilding M-tree...\n";
    auto t_mtree_start = std::chrono::high_resolution_clock::now();
    hnsw.buildMTree();
    auto t_mtree_end = std::chrono::high_resolution_clock::now();
    double mtree_sec = std::chrono::duration<double>(t_mtree_end - t_mtree_start).count();
    std::cout << "M-tree built in " << std::setprecision(2) << mtree_sec << " s.\n";
    std::cout << "MTree height: " << hnsw.getMTreeHeight() << "\n\n";

    double total_recall_hnsw = 0.0;
    double total_recall_mtree = 0.0;
    long time_hnsw_us = 0;
    long time_mtree_us = 0;

    for (int q = 0; q < Q; ++q) {
        const float* query = queries.data() + static_cast<size_t>(q) * dim;
        const int* gt_q = gt.data() + static_cast<size_t>(q) * gt_dim;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto pq_hnsw = hnsw.searchKnn(query, static_cast<size_t>(k));
        auto t1 = std::chrono::high_resolution_clock::now();
        time_hnsw_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::vector<size_t> hnsw_ids;
        hnsw_ids.reserve(k);
        while (!pq_hnsw.empty()) {
            hnsw_ids.push_back(pq_hnsw.top().second);
            pq_hnsw.pop();
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto pq_mtree = hnsw.searchKnnMTree(query, static_cast<size_t>(k));
        auto t3 = std::chrono::high_resolution_clock::now();
        time_mtree_us += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        std::vector<size_t> mtree_ids;
        mtree_ids.reserve(k);
        while (!pq_mtree.empty()) {
            mtree_ids.push_back(pq_mtree.top().second);
            pq_mtree.pop();
        }

        total_recall_hnsw += recall_at_k(gt_q, gt_dim, hnsw_ids, k);
        total_recall_mtree += recall_at_k(gt_q, gt_dim, mtree_ids, k);
    }

    float avg_recall_hnsw = static_cast<float>(total_recall_hnsw) / Q;
    float avg_recall_mtree = static_cast<float>(total_recall_mtree) / Q;
    float avg_time_hnsw_us = static_cast<float>(time_hnsw_us) / Q;
    float avg_time_mtree_us = static_cast<float>(time_mtree_us) / Q;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "SIFT1M Results (N=" << N
              << ", dim=" << dim
              << ", k=" << k
              << ", ef=" << ef_search << ")\n\n";

    std::cout << "HNSW (upper layers): Recall@" << k << " = " << avg_recall_hnsw
              << ", Avg latency (us) = " << std::fixed << std::setprecision(1) << avg_time_hnsw_us << "\n";
    std::cout << "MTree -> level-0:     Recall@" << k << " = " << std::setprecision(4) << avg_recall_mtree
              << ", Avg latency (us) = " << std::fixed << std::setprecision(1) << avg_time_mtree_us << "\n";

    std::cout << "\nSpeedup (HNSW / MTree):     " << std::fixed << std::setprecision(2)
              << (avg_time_hnsw_us / avg_time_mtree_us) << "x\n";
    std::cout << "Recall loss (HNSW - MTree): " << std::fixed << std::setprecision(4)
              << (avg_recall_hnsw - avg_recall_mtree) << "\n";

    return 0;
}
