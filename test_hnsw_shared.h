#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <iomanip>
#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"

static void read_fvecs_info(const std::string& path, int& dim, int& n) {
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
}

static std::vector<float> read_fvecs(const std::string& path, int& dim, int& n) {
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

struct HnswCache {
    std::unique_ptr<hnswlib::L2Space> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
};

static HnswCache load_or_build_hnsw(
    const std::string& data_dir,
    const std::string& serialized_index_path,
    int& dim,
    int& n,
    bool& rebuilt)
{
    const std::string base_path = data_dir + "/base.fvecs";

    int base_dim = 0;
    int base_n = 0;
    std::cout << "Building HNSW from " << base_path << "...\n";
    auto base = read_fvecs(base_path, base_dim, base_n);
    dim = base_dim;
    n = base_n;

    const int M = 16;
    const int ef_construction = 200;

    std::unique_ptr<hnswlib::L2Space> space(new hnswlib::L2Space(dim));
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw(
        new hnswlib::HierarchicalNSW<float>(space.get(), n, M, ef_construction));

    auto t_build_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        hnsw->addPoint(base.data() + static_cast<size_t>(i) * dim,
                       static_cast<size_t>(i));
        if ((i + 1) % 100000 == 0)
            std::cout << "  indexed " << (i + 1) << " / " << n << "\n";
    }
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_sec = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << "HNSW built in " << std::fixed << std::setprecision(1)
              << build_sec << " s. Max level = " << hnsw->maxlevel_ << "\n";

    rebuilt = true;
    return {std::move(space), std::move(hnsw)};
}
