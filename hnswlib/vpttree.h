#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct VPTNode;

struct VPTNode
{
    bool isLeaf = false;
    int pivot = -1;
    float median_distance = 0.0f;

    std::vector<VPTNode*> children;
    std::vector<float> childMaxDistances;

    
    std::vector<int> vectorIndices;  // for leaf nodes
    VPTNode* parent = nullptr;
    int centroidEntryPoint = -1;

    // Multi-entry information
    std::vector<std::vector<float>> leafClusterCentroids;
    std::vector<int> leafClusterRepresentatives;

    VPTNode() = default;
    VPTNode(const VPTNode&) = delete;
    VPTNode& operator=(const VPTNode&) = delete;

    ~VPTNode();
};

class VantagePointTree
{
    public:
    VantagePointTree(char* data_level0_memory, size_t data_size, size_t dim, size_t num_vectors, int leafCapacity,int numPartitions, int leafClusters = 1);

    ~VantagePointTree();

    VantagePointTree(const VantagePointTree&) = delete;
    VantagePointTree& operator=(const VantagePointTree&) = delete;

    void build();
    void recomputeLeafClusters(int leafClusters);
    int searchNN(const float* query,uint64_t& distanceCounter,int beamWidth) const;
    std::vector<int> searchNNMulti(const float* query,uint64_t& distanceCounter,int beamWidth) const;
    int getHeight() const;
    int searchEntryPoint(const float* query,uint64_t& distanceCounter,int beamWidth) const;
    std::vector<int> searchEntryPointMulti(const float* query,uint64_t& distanceCounter,int beamWidth) const;

    VPTNode* root = nullptr;
    size_t memoryUsageBytes() const;


    private:
    char* data_level0_memory;
    size_t data_size;
    size_t dim;
    size_t num_vectors;
    int leafCapacity;
    int leafClusters;
    int numPartitions;

    // Number of K-Means iterations for leaf-level clustering.
    static constexpr int MAX_KMEANS_ITERATIONS = 20;
    static constexpr float KMEANS_TOLERANCE = 1e-4f;

    const float* getVector(int id) const;
    float distance(const float* a, const float* b) const;
    VPTNode* buildRecursive(std::vector<int>&& indices);
    int calculateHeight(VPTNode* node) const;
    int choosePivot(const std::vector<int>& indices) const;
    std::vector<float> computeLeafCentroid(const VPTNode* leaf) const;
    int findNearestToCentroid(const VPTNode* leaf,const std::vector<float>& centroid) const;
    int findNearestToCentroidVec(const std::vector<float>& centroid,const std::vector<int>& vectorIndices) const;
    void preprocessLeafEntryPoints(VPTNode* node);

    std::vector<const VPTNode*> findLeafNodes(const float* query,uint64_t& distanceCounter,int beamWidth) const;
    int findNearestCentroidIdx(const float* vector,const std::vector<std::vector<float>>& centroids) const;
    void initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const;
    void runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const;
    void populateLeafMulti(VPTNode* node);
    size_t nodeMemoryUsage(const VPTNode* node) const;
};