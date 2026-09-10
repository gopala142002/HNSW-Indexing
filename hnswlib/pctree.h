#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

class PCNode 
{
    public:
    bool isLeaf;
    std::vector<float> principalComponent;
    std::vector<float> splitPoints;
    float parentDotProduct;
    std::vector<PCNode*> children;
    std::vector<int> vectorIndices;
    int representative;

    // Multi-entry information
    std::vector<std::vector<float>> leafClusterCentroids;
    std::vector<int> leafClusterRepresentatives;

    PCNode()
    {
        isLeaf = false;
        parentDotProduct = 1.0f;
        representative = -1;
    }
    ~PCNode()
    {
        for (PCNode* child : children)
            delete child;
    }
};


class PCTree 
{
    private:
        const char* data_level0_memory;
        size_t data_size;
        size_t dim;
        size_t num_vectors;
        int leafCapacity;
        int numPartitions;
        int leafClusters;
        PCNode* root;

        // Number of K-Means iterations for leaf-level clustering.
        static constexpr int MAX_KMEANS_ITERATIONS = 20;
        static constexpr float KMEANS_TOLERANCE = 1e-4f;

        inline const float* getVector(int id) const 
        {
            return reinterpret_cast<const float*>(data_level0_memory + (size_t)id * data_size);
        }
        
        void computePCA(const std::vector<int>& vectorIndices,std::vector<float>& pc);
        
        std::vector<float> computeCentroid(const std::vector<int>& vectorIndices);
    
        int findNearestToPoint(const float* point,const std::vector<int>& vectorIndices);
    
        PCNode* buildNode(std::vector<int> &vectorIndices,const std::vector<float>& parentPC);

        void populateLeaf(PCNode* node);
        int findNearestCentroid(const float* vector,const std::vector<std::vector<float>>& centroids) const;
        void initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const;
        void runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const;
        PCNode* findLeaf(const float* query) const;

        inline float dotProduct(const float* a, const float* b) const 
        {
            float result = 0.0f;
            for (size_t i = 0; i < dim; ++i) 
            {
                result += a[i] * b[i];
            }
            return result;
        }

        inline float squaredL2Distance(const float* a, const float* b) const 
        {
            float result = 0.0f;
            for (size_t i = 0; i < dim; ++i) 
            {
                float diff = a[i] - b[i];
                result += diff * diff;
            }
            return result;
        }
    public:
        PCTree(const char* data_level0_memory, size_t data_size, size_t dim,size_t num_vectors, int leafCapacity,int numPartitions,int leafClusters);
        ~PCTree();
        std::vector<int> searchNN(const float* query) const;
        std::vector<int> searchNNMulti(const float* query) const;
        int getHeight(PCNode* root) const;
        int getHeight() const;
};