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

#include "test_hnsw_shared.h"
#include "hnswlib/hnswlib.h"

std::vector<int> read_ivecs(
    const std::string& path,
    int& dim,
    int& n)
{
    std::ifstream f(path, std::ios::binary);

    if (!f)
        throw std::runtime_error("Cannot open " + path);

    int32_t d;

    f.read(
        reinterpret_cast<char*>(&d),
        4);

    dim = d;

    f.seekg(0, std::ios::end);
    std::streamsize fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    const std::streamsize record =
        4 + static_cast<std::streamsize>(d) * 4;

    n = static_cast<int>(fsize / record);

    std::vector<int> data(
        static_cast<size_t>(n) * d);

    for (int i = 0; i < n; ++i)
    {
        int32_t dd;

        f.read(
            reinterpret_cast<char*>(&dd),
            4);

        if (dd != d)
            throw std::runtime_error(
                "Inconsistent ivecs dimension");

        f.read(
            reinterpret_cast<char*>(
                &data[static_cast<size_t>(i) * d]),
            4 * d);
    }

    return data;
}

float recall_at_k(
    const int* gt,
    int gt_k,
    const std::vector<size_t>& returned,
    int k)
{
    const int effective_k =
        std::min(gt_k, k);

    std::unordered_set<int> gt_set(
        gt,
        gt + effective_k);

    int hits = 0;

    for (size_t id : returned)
    {
        if (gt_set.count(static_cast<int>(id)))
            ++hits;
    }

    return static_cast<float>(hits) /
           static_cast<float>(effective_k);
}

std::vector<size_t> extract_ids(
    std::priority_queue<
        std::pair<float, hnswlib::labeltype>>& pq)
{
    std::vector<size_t> ids;
    ids.reserve(pq.size());

    while (!pq.empty())
    {
        ids.push_back(
            static_cast<size_t>(pq.top().second));

        pq.pop();
    }

    return ids;
}

int main(int argc, char** argv)
{
    const std::string data_dir =
        (argc > 1)
            ? argv[1]
            : "./datasets/gist";

    const std::string base_path =
        data_dir + "/base.fvecs";

    const std::string query_path =
        data_dir + "/query.fvecs";

    const std::string gt_path =
        data_dir + "/groundtruth.ivecs";

    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 50;
    const int k = 30;
    const int num_queries = 10000;

    const int num_pivots = 2;
    const int leaf_cap = 64;

    const int finger_rank = 16;
    const int finger_warmup = 4;

    std::cout
        << "Loading dataset base vectors from "
        << base_path
        << "\n";

    int base_dim;
    int base_n;

    auto base =
        read_fvecs(
            base_path,
            base_dim,
            base_n);

    std::cout
        << "base: "
        << base_n
        << " x "
        << base_dim
        << "\n";

    std::cout
        << "Loading queries from "
        << query_path
        << "\n";

    int q_dim;
    int q_n;

    auto queries =
        read_fvecs(
            query_path,
            q_dim,
            q_n);

    std::cout
        << "queries: "
        << q_n
        << " x "
        << q_dim
        << "\n";

    if (base_dim != q_dim)
    {
        throw std::runtime_error(
            "Dimension mismatch between base and queries");
    }

    std::cout
        << "Loading ground truth from "
        << gt_path
        << " ...\n";

    int gt_dim;
    int gt_n;

    auto gt =
        read_ivecs(
            gt_path,
            gt_dim,
            gt_n);

    std::cout
        << "gt: "
        << gt_n
        << " x "
        << gt_dim
        << "\n\n";

    const int Q =
        std::min(num_queries, q_n);

    const std::string cache_path =
        data_dir + "/gist_hnsw.bin";

    bool rebuilt = false;

    HnswCache cache =
        load_or_build_hnsw(
            data_dir,
            cache_path,
            base_dim,
            base_n,
            rebuilt);

    auto& hnsw = *cache.index;

    const int dim = base_dim;
    const int N = base_n;

    hnsw.setEf(ef_search);

    std::cout
        << "\nBuilding VoronoiTree "
        << "(numPivots="
        << num_pivots
        << ", leafCapacity="
        << leaf_cap
        << ")...\n";

    auto start =
        std::chrono::high_resolution_clock::now();

    hnsw.buildVoronoiTree();

    auto end =
        std::chrono::high_resolution_clock::now();

    double vtree_build_time =
        std::chrono::duration<double>(
            end - start).count();

    std::cout
        << "VoronoiTree built in "
        << std::fixed
        << std::setprecision(2)
        << vtree_build_time
        << " s.\n";

    std::cout
        << "VoronoiTree height: "
        << hnsw.getVTreeHeight()
        << "\n";

    std::cout
        << "\nBuilding FINGER structures...\n";

    hnsw.setFingerRank(finger_rank);
    hnsw.setFingerWarmup(finger_warmup);
    hnsw.setUseFinger(true, finger_rank);

    auto finger_start =
        std::chrono::high_resolution_clock::now();

    hnsw.buildFingerProjections();

    auto finger_end =
        std::chrono::high_resolution_clock::now();

    double finger_time =
        std::chrono::duration<double>(
            finger_end - finger_start).count();

    std::cout
        << "FINGER built in "
        << std::fixed
        << std::setprecision(2)
        << finger_time
        << " s.\n";

    std::cout
        << "FINGER rank: "
        << hnsw.getFingerRank()
        << "\n";

    std::cout
        << "FINGER warmup: "
        << hnsw.getFingerWarmup()
        << "\n\n";

    double total_recall_hnsw = 0.0;
    double total_recall_vtree = 0.0;
    double total_recall_vtree_finger = 0.0;

    long long total_time_hnsw_us = 0;
    long long total_time_vtree_us = 0;
    long long total_time_vtree_finger_us = 0;

    std::cout
        << "Running "
        << Q
        << " queries (k="
        << k
        << ", ef="
        << ef_search
        << ")...\n\n";

    for (int q = 0; q < Q; ++q)
    {
        const float* query =
            queries.data() +
            static_cast<size_t>(q) * dim;

        const int* gt_q =
            gt.data() +
            static_cast<size_t>(q) * gt_dim;

        auto t0 =
            std::chrono::high_resolution_clock::now();

        auto pq_hnsw =
            hnsw.searchKnn(
                query,
                static_cast<size_t>(k));

        auto t1 =
            std::chrono::high_resolution_clock::now();

        total_time_hnsw_us +=
            std::chrono::duration_cast<
                std::chrono::microseconds>(
                t1 - t0).count();

        auto hnsw_ids =
            extract_ids(pq_hnsw);

        total_recall_hnsw +=
            recall_at_k(
                gt_q,
                gt_dim,
                hnsw_ids,
                k);

        auto t2 =
            std::chrono::high_resolution_clock::now();

        auto pq_vtree =
            hnsw.searchKnnVTree(
                query,
                static_cast<size_t>(k));

        auto t3 =
            std::chrono::high_resolution_clock::now();

        total_time_vtree_us +=
            std::chrono::duration_cast<
                std::chrono::microseconds>(
                t3 - t2).count();

        auto vtree_ids =
            extract_ids(pq_vtree);

        total_recall_vtree +=
            recall_at_k(
                gt_q,
                gt_dim,
                vtree_ids,
                k);

        auto t4 =
            std::chrono::high_resolution_clock::now();

        auto pq_vtree_finger =
            hnsw.searchKnnVTreeFinger(
                query,
                static_cast<size_t>(k));

        auto t5 =
            std::chrono::high_resolution_clock::now();

        total_time_vtree_finger_us +=
            std::chrono::duration_cast<
                std::chrono::microseconds>(
                t5 - t4).count();

        auto vtree_finger_ids =
            extract_ids(pq_vtree_finger);

        total_recall_vtree_finger +=
            recall_at_k(
                gt_q,
                gt_dim,
                vtree_finger_ids,
                k);

        if ((q + 1) % 1000 == 0)
        {
            std::cout
                << "completed "
                << q + 1
                << " / "
                << Q
                << "\n";
        }
    }

    const double avg_recall_hnsw =
        total_recall_hnsw / Q;

    const double avg_recall_vtree =
        total_recall_vtree / Q;

    const double avg_recall_vtree_finger =
        total_recall_vtree_finger / Q;

    const double avg_time_hnsw_us =
        static_cast<double>(
            total_time_hnsw_us) / Q;

    const double avg_time_vtree_us =
        static_cast<double>(
            total_time_vtree_us) / Q;

    const double avg_time_vtree_finger_us =
        static_cast<double>(
            total_time_vtree_finger_us) / Q;

    const double speedup_vtree =
        avg_time_hnsw_us /
        avg_time_vtree_us;

    const double speedup_vtree_finger =
        avg_time_hnsw_us /
        avg_time_vtree_finger_us;

    const double speedup_finger_over_vtree =
        avg_time_vtree_us /
        avg_time_vtree_finger_us;

    

    std::cout
        << std::left
        << std::setw(30)
        << "Method"
        << std::right
        << std::setw(18)
        << "Avg Time (us)"
        << std::setw(15)
        << "Recall@"
        << k
        << std::setw(15)
        << "Speedup"
        << "\n";


    std::cout
        << std::left
        << std::setw(30)
        << "Normal HNSW"
        << std::right
        << std::setw(18)
        << std::fixed
        << std::setprecision(2)
        << avg_time_hnsw_us
        << std::setw(15)
        << std::setprecision(4)
        << avg_recall_hnsw
        << std::setw(15)
        << std::setprecision(2)
        << "1.00x"
        << "\n";

    std::cout
        << std::left
        << std::setw(30)
        << "VTree -> Normal HNSW"
        << std::right
        << std::setw(18)
        << std::fixed
        << std::setprecision(2)
        << avg_time_vtree_us
        << std::setw(15)
        << std::setprecision(4)
        << avg_recall_vtree
        << std::setw(15)
        << std::setprecision(2)
        << speedup_vtree
        << "x\n";

    std::cout
        << std::left
        << std::setw(30)
        << "VTree -> HNSW-FINGER"
        << std::right
        << std::setw(18)
        << std::fixed
        << std::setprecision(2)
        << avg_time_vtree_finger_us
        << std::setw(15)
        << std::setprecision(4)
        << avg_recall_vtree_finger
        << std::setw(15)
        << std::setprecision(2)
        << speedup_vtree_finger
        << "x\n";

    std::cout
        << "\nSpeedup VTree+FINGER / VTree = "
        << std::fixed
        << std::setprecision(2)
        << speedup_finger_over_vtree
        << "x\n";

    std::cout
        << "Recall loss HNSW - VTree = "
        << std::setprecision(4)
        << avg_recall_hnsw -
           avg_recall_vtree
        << "\n";

    std::cout
        << "Recall loss HNSW - VTree+FINGER = "
        << avg_recall_hnsw -
           avg_recall_vtree_finger
        << "\n";

    std::cout
        << "Recall difference VTree - "
        << "VTree+FINGER = "
        << avg_recall_vtree -
           avg_recall_vtree_finger
        << "\n";

   
    return 0;
}