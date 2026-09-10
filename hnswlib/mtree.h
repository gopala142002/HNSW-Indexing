#pragma once
#define MTREE_H

#include <cstddef>
#include <utility>
#include <vector>

class MTNode; 

class RoutingEntry 
{
    public:
    int pivotID = -1;
    MTNode* child = nullptr;
};

class ObjectEntry 
{
    public:
    int vectorID = -1;
};

class MTNode
{
    public:
    bool isLeaf = false;
    std::vector<RoutingEntry> routingEntries;
    std::vector<ObjectEntry> objectEntries;

    int centroidEntryPoint = -1;

    // Multi-entry information
    std::vector<std::vector<float>> leafClusterCentroids;
    std::vector<int> leafClusterRepresentatives;

    MTNode() = default;
    MTNode(const MTNode&) = delete;
    MTNode& operator=(const MTNode&) = delete;
    ~MTNode();
};

class MTree 
{
    public:
        MTree(char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int maxRoutingEntries = 8,int maxLeafObjects = 32,int leafClusters = 1);

        MTree(char* data_level0_memory,size_t data_size,size_t vector_offset,size_t dim,size_t num_vectors,int maxRoutingEntries,int maxLeafObjects,int leafClusters);

        ~MTree();

        MTree(const MTree&) = delete;
        MTree& operator=(const MTree&) = delete;

        void build();
        void insert(int vectorID);

        int greedySearch(const float* query) const;
        int searchEntryPoint(const float* query) const;
        std::vector<int> searchEntryPointMulti(const float* query) const;
        int getHeight() const;

        MTNode* root = nullptr;

    private:
        char* data_level0_memory;
        size_t data_size;
        size_t vector_offset;
        size_t dim;
        size_t num_vectors;

        int maxRoutingEntries;
        int maxLeafObjects;
        int leafClusters;

        // Number of K-Means iterations for leaf-level clustering.
        static constexpr int MAX_KMEANS_ITERATIONS = 20;
        static constexpr float KMEANS_TOLERANCE = 1e-4f;

        struct InsertResult 
        {
            bool split = false;
            std::vector<RoutingEntry> entries;
        };

        const float* getVector(int id) const;
        float distance(const float* a, const float* b) const;
        float distanceToId(int a, int b) const;

        void insertObject(int vectorID);
        InsertResult insertIntoNode(MTNode* node, int vectorID);
        InsertResult splitNode(MTNode* node);

        std::vector<int> collectObjectIds(const MTNode* node) const;
        int calculateHeight(const MTNode* node) const;
        std::vector<int> collectRoutingIds(const MTNode* node) const;
        std::pair<int, int> selectPromoters(const std::vector<int>& ids) const;
        int assignToPromoter(int id, const std::vector<int>& promoters) const;
        std::vector<float> computeLeafCentroid(const MTNode* leaf) const;
        int findNearestToCentroid(const MTNode* leaf,const std::vector<float>& centroid) const;
        int findNearestToCentroidVec(const std::vector<float>& centroid,const std::vector<int>& vectorIndices) const;
        void preprocessLeafEntryPoints(MTNode* node);

        const MTNode* findLeafNode(const float* query) const;
        int findNearestCentroidIdx(const float* vector,const std::vector<std::vector<float>>& centroids) const;
        void initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const;
        void runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const;
        void populateLeafMulti(MTNode* node);
};