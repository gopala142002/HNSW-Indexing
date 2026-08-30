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
    const int num_pivots = 2;



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
    std::cout<< "\nBuilding PCTree "<< "(leafCapacity="<< leaf_capacity<< ")...\n";
    auto t_pc_start =std::chrono::high_resolution_clock::now();
    hnsw.buildPCTree();
    auto t_pc_end =std::chrono::high_resolution_clock::now();
    double pc_sec =std::chrono::duration<double>(t_pc_end - t_pc_start).count();
    std::cout<< "PCTree built in "<< std::setprecision(2)<< pc_sec<< " s.\n";
    const int pctree_height = hnsw.getPCTreeHeight();
    std::cout<< "PCTree height: "<< pctree_height<< "\n";



    // Building MTree
    std::cout<< "\nBuilding MTree "<< "(leafCapacity="<< leaf_capacity<< ")...\n";
    auto t_mtree_start =std::chrono::high_resolution_clock::now();
    hnsw.buildMTree();
    auto t_mtree_end =std::chrono::high_resolution_clock::now();
    double mtree_sec =std::chrono::duration<double>(t_mtree_end - t_mtree_start).count();
    std::cout<< "MTree built in "<< std::setprecision(2)<< mtree_sec<< " s.\n";
    const int mtree_height = hnsw.getMTreeHeight();
    std::cout<< "MTree height: "<< mtree_height<< "\n";




    // Building VPTree
    std::cout<< "\nBuilding VantagePointTree "<< "(leafCapacity="<< leaf_capacity<< ")...\n";
    auto t_vp_start =std::chrono::high_resolution_clock::now();
    hnsw.buildVantagePointTree();
    auto t_vp_end =std::chrono::high_resolution_clock::now();
    double vpt_build_time =std::chrono::duration<double>(t_vp_end - t_vp_start).count();
    std::cout<< "VantagePointTree built in "<< std::fixed<< std::setprecision(2)<< vpt_build_time<< " s.\n";
    const int vptree_height = hnsw.getVPTreeHeight();
    std::cout<< "VantagePointTree height: "<< vptree_height << "\n";


    // Building VTree
    std::cout<< "\nBuilding VoronoiTree "<< "(numPivots="<< num_pivots<< ", leafCapacity="<< leaf_capacity<< ")...\n";
    auto start =std::chrono::high_resolution_clock::now();
    hnsw.buildVoronoiTree();
    auto end =std::chrono::high_resolution_clock::now();
    double vtree_build_time =std::chrono::duration<double>(end - start).count();
    std::cout<< "VoronoiTree built in "<< std::fixed<< std::setprecision(2)<< vtree_build_time<< " s.\n";
    const int vtree_height = hnsw.getVTreeHeight();
    std::cout<< "VoronoiTree height: "<< vtree_height << "\n";




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
    std::cout<< "Running "<< Q<< " queries (k="<< k<< ", ef="<< ef_search<< ")...\n\n";



    double total_recall_hnsw = 0.0;
    double total_recall_hnsw_finger = 0.0;
    double total_recall_pctree = 0.0;
    double total_recall_pctree_finger = 0.0;
    double total_recall_mtree = 0.0;
    double total_recall_mtree_finger = 0.0;
    double total_recall_vpt=0.0;
    double total_recall_vpt_finger = 0.0;
    double total_recall_vtree = 0.0;
    double total_recall_vtree_finger = 0.0;


    // Accumulate nanoseconds as double so sub-microsecond measurements are not
    // truncated to zero before averaging.
    double time_hnsw_ns = 0.0;
    double time_hnsw_finger_ns = 0.0;
    double time_pctree_ns = 0.0;
    double time_pctree_finger_ns = 0.0;
    double time_mtree_ns = 0.0;
    double time_mtree_finger_ns = 0.0;
    double time_vpt_ns = 0.0;
    double time_vpt_finger_ns = 0.0;
    double time_vtree_ns = 0.0;
    double time_vtree_finger_ns = 0.0;



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


        // for mtree->(level-0 + finer optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_mtree_finger =hnsw.searchKnnMTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_mtree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto mtree_finger_ids =extract_ids(pq_mtree_finger,k);
        total_recall_mtree_finger +=recall_at_k(gt_q,gt_dim,mtree_finger_ids,k);



        // for pctree -> hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_pctree =hnsw.searchKnnPCTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_pctree_ns += std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> pctree_ids =extract_ids(pq_pctree, k);
        total_recall_pctree +=recall_at_k(gt_q,gt_dim,pctree_ids,k);
        


        // for pctree -> hnsw(level-0 + finger optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_pctree_finger =hnsw.searchKnnPCTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_pctree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        std::vector<size_t> pctree_finger_ids =extract_ids(pq_pctree_finger,k);
        total_recall_pctree_finger +=recall_at_k(gt_q,gt_dim,pctree_finger_ids,k);



        // for vptree->hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vpt =hnsw.searchKnnVPTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vpt_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_ids =extract_ids(pq_vpt, k);
        total_recall_vpt +=recall_at_k(gt_q,gt_dim,vpt_ids,k);



        // for vptree->(level-0 + finer optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vpt_finger =hnsw.searchKnnVPTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vpt_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vpt_finger_ids =extract_ids(pq_vpt_finger, k);
        total_recall_vpt_finger +=recall_at_k(gt_q,gt_dim,vpt_finger_ids,k);



        // for vtree->hnsw(level-0)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vtree =hnsw.searchKnnVTree(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vtree_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vtree_ids =extract_ids(pq_vtree, k);
        total_recall_vtree +=recall_at_k(gt_q,gt_dim,vtree_ids,k);



        // for vtree->(level-0 + finer optimization)
        start =std::chrono::high_resolution_clock::now();
        auto pq_vtree_finger =hnsw.searchKnnVTreeFinger(query,static_cast<size_t>(k));
        end =std::chrono::high_resolution_clock::now();
        time_vtree_finger_ns += std::chrono::duration<double, std::nano>(end - start).count();
        auto vtree_finger_ids =extract_ids(pq_vtree_finger,k);
        total_recall_vtree_finger +=recall_at_k(gt_q,gt_dim,vtree_finger_ids,k);
        


        if ((q + 1) % 1000 == 0) 
        {
            std::cout<< "  completed "<< (q + 1)<< " / "<< Q<< "\n";
        }
    }


    const double avg_recall_hnsw = total_recall_hnsw / Q;
    const double avg_recall_hnsw_finger =total_recall_hnsw_finger / Q;
    const double avg_recall_pctree = total_recall_pctree / Q;
    const double avg_recall_pctree_finger = total_recall_pctree_finger / Q;
    const double avg_recall_mtree = total_recall_mtree / Q;
    const double avg_recall_mtree_finger = total_recall_mtree_finger / Q;
    const double avg_recall_vpt = total_recall_vpt / Q;
    const double avg_recall_vpt_finger = total_recall_vpt_finger / Q;
    const double avg_recall_vtree = total_recall_vtree / Q;
    const double avg_recall_vtree_finger = total_recall_vtree_finger / Q;

    const double avg_time_hnsw_us = (time_hnsw_ns / Q) / 1000.0;
    const double avg_time_hnsw_finger_us = (time_hnsw_finger_ns / Q) / 1000.0;
    const double avg_time_pctree_us = (time_pctree_ns / Q) / 1000.0;
    const double avg_time_pctree_finger_us = (time_pctree_finger_ns / Q) / 1000.0;
    const double avg_time_mtree_us = (time_mtree_ns / Q) / 1000.0;
    const double avg_time_mtree_finger_us = (time_mtree_finger_ns / Q) / 1000.0;
    const double avg_time_vpt_us = (time_vpt_ns / Q) / 1000.0;
    const double avg_time_vpt_finger_us = (time_vpt_finger_ns / Q) / 1000.0;
    const double avg_time_vtree_us = (time_vtree_ns / Q) / 1000.0;
    const double avg_time_vtree_finger_us = (time_vtree_finger_ns / Q) / 1000.0;

    const double speedup_pctree = avg_time_hnsw_us / avg_time_pctree_us;
    const double speedup_hnsw_finger = avg_time_hnsw_us / avg_time_hnsw_finger_us;
    const double speedup_pctree_finger = avg_time_hnsw_us / avg_time_pctree_finger_us;
    const double speedup_mtree = avg_time_hnsw_us / avg_time_mtree_us;
    const double speedup_mtree_finger = avg_time_hnsw_us / avg_time_mtree_finger_us;
    const double speedup_vpt = avg_time_hnsw_us / avg_time_vpt_us;
    const double speedup_vpt_finger = avg_time_hnsw_us / avg_time_vpt_finger_us;
    const double speedup_vtree = avg_time_hnsw_us / avg_time_vtree_us;
    const double speedup_vtree_finger = avg_time_hnsw_us / avg_time_vtree_finger_us;

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



    print_result("Normal HNSW", avg_recall_hnsw, avg_time_hnsw_us, 1.0);
    print_result("HNSW + FINGER",avg_recall_hnsw_finger,avg_time_hnsw_finger_us,speedup_hnsw_finger);
    print_result("MTree -> HNSW", avg_recall_mtree, avg_time_mtree_us, speedup_mtree);
    print_result("MTree -> FINGER", avg_recall_mtree_finger,avg_time_mtree_finger_us, speedup_mtree_finger);
    print_result("PCTree -> HNSW", avg_recall_pctree, avg_time_pctree_us, speedup_pctree);
    print_result("PCTree -> FINGER", avg_recall_pctree_finger,avg_time_pctree_finger_us, speedup_pctree_finger);
    print_result("VPTree -> HNSW", avg_recall_vpt, avg_time_vpt_us, speedup_vpt);
    print_result("VPTree -> FINGER", avg_recall_vpt_finger,avg_time_vpt_finger_us, speedup_vpt_finger);
    print_result("VTree -> HNSW", avg_recall_vtree, avg_time_vtree_us, speedup_vtree);
    print_result("VTree -> FINGER", avg_recall_vtree_finger,avg_time_vtree_finger_us, speedup_vtree_finger);



    // One summary row per search method. This file is overwritten on every run.
    const std::string csv_path = "hnsw_results_sift.csv";
    std::ofstream csv(csv_path);
    if (!csv) 
    {
        throw std::runtime_error("Cannot open " + csv_path + " for writing");
    }

    csv << "method,N,dim,k,ef_search,num_queries,M,ef_construction,"
       "leaf_capacity,finger_rank,finger_warmup,num_pivots,"
       "hnsw_num_levels,pctree_height,mtree_height,vptree_height,vtree_height,"
       "hnsw_build_sec,pctree_build_sec,mtree_build_sec,vptree_build_sec,"
       "vtree_build_sec,finger_build_sec,recall_at_k,avg_latency_us,"
       "speedup_vs_hnsw,recall_loss_vs_hnsw\n";

    auto write_row = [&](const std::string& method, double recall,double latency_us, double speedup) 
    {
        csv << std::setprecision(10)
            << method << ','
            << N << ',' << dim << ',' << k << ',' << ef_search << ',' << Q << ','
            << M << ',' << ef_construction << ',' << leaf_capacity << ','
            << finger_rank << ',' << finger_warmup << ',' << num_pivots << ','
            << hnsw_num_levels << ','
            << pctree_height << ','
            << mtree_height << ','
            << vptree_height << ','
            << vtree_height << ','
            << build_sec << ',' << pc_sec << ',' << mtree_sec << ','
            << vpt_build_time << ',' << vtree_build_time << ',' << finger_build_sec << ','
            << recall << ',' << latency_us << ',' << speedup << ','
            << (avg_recall_hnsw - recall) << '\n';
    };

    write_row("Normal HNSW", avg_recall_hnsw, avg_time_hnsw_us, 1.0);
    write_row("HNSW + FINGER",avg_recall_hnsw_finger,avg_time_hnsw_finger_us,speedup_hnsw_finger);
    write_row("MTree -> HNSW", avg_recall_mtree, avg_time_mtree_us, speedup_mtree);
    write_row("MTree -> FINGER", avg_recall_mtree_finger,avg_time_mtree_finger_us, speedup_mtree_finger);
    write_row("PCTree -> HNSW", avg_recall_pctree, avg_time_pctree_us, speedup_pctree);
    write_row("PCTree -> FINGER", avg_recall_pctree_finger,avg_time_pctree_finger_us, speedup_pctree_finger);
    write_row("VPTree -> HNSW", avg_recall_vpt, avg_time_vpt_us, speedup_vpt);
    write_row("VPTree -> FINGER", avg_recall_vpt_finger,avg_time_vpt_finger_us, speedup_vpt_finger);
    write_row("VTree -> HNSW", avg_recall_vtree, avg_time_vtree_us, speedup_vtree);
    write_row("VTree -> FINGER", avg_recall_vtree_finger,avg_time_vtree_finger_us, speedup_vtree_finger);

    csv.close();

    std::cout << "\nCSV written to: " << csv_path << "\n";
    return 0;
}