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
    const std::string data_dir =(argc > 1)? argv[1]: "./datasets/gist";
    const std::string base_path =data_dir + "/base.fvecs";
    const std::string query_path =data_dir + "/query.fvecs";
    const std::string gt_path =data_dir + "/groundtruth.ivecs";


    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 50;
    const int k = 30;
    const int num_queries = 10000;
    const int leaf_capacity = 64;
    const int finger_rank = 16;
    const int finger_warmup = 8;



    std::cout<< "Loading SIFT1M base vectors from "<< base_path << " ...\n";
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

    std::cout<< "\nBuilding PCTree "<< "(leafCapacity="<< leaf_capacity<< ")...\n";

    auto t_pc_start =std::chrono::high_resolution_clock::now();

    hnsw.buildPCTree();

    auto t_pc_end =std::chrono::high_resolution_clock::now();

    double pc_sec =std::chrono::duration<double>(t_pc_end - t_pc_start).count();

    std::cout<< "PCTree built in "<< std::setprecision(2)<< pc_sec<< " s.\n";

    std::cout<< "PCTree height: "<< hnsw.getPCTreeHeight()<< "\n";

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

    std::cout<< "Running "<< Q<< " queries (k="<< k<< ", ef="<< ef_search<< ")...\n\n";

    double total_recall_hnsw = 0.0;
    double total_recall_pctree = 0.0;
    double total_recall_pctree_finger = 0.0;

    long long time_hnsw_us = 0;
    long long time_pctree_us = 0;
    long long time_pctree_finger_us = 0;

    for (int q = 0; q < Q; ++q) 
    {
        const float* query =queries.data() +static_cast<size_t>(q) * dim;

        const int* gt_q =gt.data() +static_cast<size_t>(q) * gt_dim;

        auto t0 =std::chrono::high_resolution_clock::now();

        auto pq_hnsw =hnsw.searchKnn(query,static_cast<size_t>(k));

        auto t1 =std::chrono::high_resolution_clock::now();

        time_hnsw_us +=std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        std::vector<size_t> hnsw_ids =extract_ids(pq_hnsw, k);

        auto t2 =std::chrono::high_resolution_clock::now();

        auto pq_pctree =hnsw.searchKnnPCTree(query,static_cast<size_t>(k));

        auto t3 =std::chrono::high_resolution_clock::now();

        time_pctree_us +=std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        std::vector<size_t> pctree_ids =extract_ids(pq_pctree, k);

        auto t4 =std::chrono::high_resolution_clock::now();
        auto pq_pctree_finger =hnsw.searchKnnPCTreeFinger(query,static_cast<size_t>(k));
        auto t5 =std::chrono::high_resolution_clock::now();

        time_pctree_finger_us +=std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

        std::vector<size_t> pctree_finger_ids =extract_ids(pq_pctree_finger,k);

        total_recall_hnsw +=recall_at_k(gt_q,gt_dim,hnsw_ids,k);
        total_recall_pctree +=recall_at_k(gt_q,gt_dim,pctree_ids,k);

        total_recall_pctree_finger +=recall_at_k(gt_q,gt_dim,pctree_finger_ids,k);

        if ((q + 1) % 1000 == 0) 
        {
            std::cout<< "  completed "<< (q + 1)<< " / "<< Q<< "\n";
        }
    }

    const float avg_recall_hnsw =static_cast<float>(total_recall_hnsw / Q);

    const float avg_recall_pctree =
        static_cast<float>(
            total_recall_pctree / Q
        );

    const float avg_recall_pctree_finger =
        static_cast<float>(
            total_recall_pctree_finger / Q
        );

    const double avg_time_hnsw_us =
        static_cast<double>(
            time_hnsw_us
        ) / Q;

    const double avg_time_pctree_us =
        static_cast<double>(
            time_pctree_us
        ) / Q;

    const double avg_time_pctree_finger_us =
        static_cast<double>(
            time_pctree_finger_us
        ) / Q;

   

    std::cout<< "SIFT1M RESULTS\n";

    std::cout<< "N=" << N<< ", dim=" << dim<< ", k=" << k<< ", ef=" << ef_search<< "\n";

   

    std::cout<< std::fixed<< std::setprecision(4);

    std::cout<< "Normal HNSW\n"<< "Recall@" << k<< " = "<< avg_recall_hnsw<< "\n"<< "Avg latency = "<< std::setprecision(1)<< avg_time_hnsw_us<< " us\n\n";

    std::cout<< "PCTree -> Normal Level-0\n"<< "Recall@" << k<< " = "<< std::setprecision(4)<< avg_recall_pctree<< "\n"<< "Avg latency = "<< std::setprecision(1)<< avg_time_pctree_us<< " us\n\n";

    std::cout<< "PCTree -> FINGER Level-0\n"<< "Recall@" << k<< " = "<< std::setprecision(4)<< avg_recall_pctree_finger<< "\n"<< "Avg latency = "<< std::setprecision(1)<< avg_time_pctree_finger_us<< " us\n\n";

    std::cout<< "Speedup PCTree / HNSW = "<< std::setprecision(2)<< avg_time_hnsw_us /avg_time_pctree_us<< "x\n";
    std::cout<< "Speedup PCTree+FINGER / HNSW = "<< avg_time_hnsw_us /avg_time_pctree_finger_us<< "x\n";
    std::cout<< "Speedup PCTree+FINGER / PCTree = "<< avg_time_pctree_us /avg_time_pctree_finger_us<< "x\n";
    std::cout<< "Recall loss HNSW - PCTree = "<< avg_recall_hnsw -avg_recall_pctree<< "\n";
    std::cout<< "Recall loss HNSW - PCTree+FINGER = "<< avg_recall_hnsw -avg_recall_pctree_finger<< "\n";
    std::cout<< "Recall difference PCTree - "<< "PCTree+FINGER = "<< avg_recall_pctree -avg_recall_pctree_finger<< "\n";
    return 0;
}