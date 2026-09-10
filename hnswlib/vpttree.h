#pragma once
#define VPTTREE_H
#include <vector>

struct VPTNode;

struct VPTNode
{
    bool isLeaf = false;
    int pivot = -1;
    float median_distance = 0.0f;
    VPTNode* left = nullptr;   // points with distance <= median
    VPTNode* right = nullptr;  // points with distance > median
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
    VantagePointTree(char* data_level0_memory, size_t data_size, size_t dim, size_t num_vectors, int leafCapacity, int leafClusters = 1);

    ~VantagePointTree();

    VantagePointTree(const VantagePointTree&) = delete;
    VantagePointTree& operator=(const VantagePointTree&) = delete;

    void build();

    std::vector<int> searchNN(const float* query) const;
    int getHeight() const;
    int searchEntryPoint(const float* query) const;
    std::vector<int> searchEntryPointMulti(const float* query) const;

    VPTNode* root = nullptr;
    private:
    char* data_level0_memory;
    size_t data_size;
    size_t dim;
    size_t num_vectors;
    int leafCapacity;
    int leafClusters;

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

    const VPTNode* findLeafNode(const float* query) const;
    int findNearestCentroidIdx(const float* vector,const std::vector<std::vector<float>>& centroids) const;
    void initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const;
    void runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const;
    void populateLeafMulti(VPTNode* node);
};