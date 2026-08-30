#pragma once
#include <vector>
#include <cstddef>
class KMeansNode
{
    public:
    bool isLeaf;
    std::vector<std::vector<float>> centroids;
    std::vector<KMeansNode*> children;
    std::vector<int> vectorIndices;

    int representative;
    KMeansNode()
    {
        isLeaf = false;
        representative = -1;
    }

    KMeansNode(const KMeansNode&) = delete;
    KMeansNode& operator=(const KMeansNode&) = delete;

    ~KMeansNode()
    {
        for (KMeansNode* child : children)
        {
            delete child;
        }
    }
};


class KMeansTree
{
public:

    KMeansTree(
        const char* data_level0_memory,
        size_t data_size,
        size_t dim,
        size_t num_vectors,
        int numClusters,
        int leafCapacity
    );

    ~KMeansTree();

    KMeansTree(const KMeansTree&) = delete;
    KMeansTree& operator=(const KMeansTree&) = delete;

    void build();

    int searchNN(const float* query) const;

    int getHeight() const;

    KMeansNode* root = nullptr;
    
    private:

    const char* data_level0_memory;

    size_t data_size;
    size_t dim;
    size_t num_vectors;

    int numClusters;
    int leafCapacity;

    // Number of K-Means iterations at each node.
    static constexpr int MAX_KMEANS_ITERATIONS = 20;
    static constexpr float KMEANS_TOLERANCE = 1e-4f;

    inline const float* getVector(int id) const
    {
        return reinterpret_cast<const float*>(data_level0_memory +static_cast<size_t>(id) * data_size);
    }

    float squaredDistance(const float* a,const float* b) const;
    float squaredDistance(const std::vector<float>& a,const float* b) const;
    int findNearestCentroid(const float* vector,const std::vector<std::vector<float>>& centroids) const;
    void initializeCentroids(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids) const;
    void runKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const;
    int findNearestToPoint(const std::vector<float>& point,const std::vector<int>& vectorIndices) const;
    KMeansNode* buildRecursive(std::vector<int>&& vectorIndices);
    int calculateHeight(KMeansNode* node) const;
};