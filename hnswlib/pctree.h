#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

class PCNode 
{
    public:
    bool isLeaf;    
    std::vector<float> principalComponent;
    float medianProjection;
    float parentDotProduct;
    PCNode* left;
    PCNode* right;
    std::vector<int> vectorIndices;
    int representative;
    
    PCNode()
    {
        isLeaf=false;
        medianProjection=0.0f;
        parentDotProduct=1.0f;
        left=nullptr;
        right=nullptr;
        representative=-1;
    }

    ~PCNode() 
    {
        delete left;
        delete right;
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
        
        PCNode* root;

        inline const float* getVector(int id) const 
        {
            return reinterpret_cast<const float*>(data_level0_memory + (size_t)id * data_size);
        }
        
        void computePCA(const std::vector<int>& vectorIndices,std::vector<float>& pc);
        
        std::vector<float> computeCentroid(const std::vector<int>& vectorIndices);
    
        int findNearestToPoint(const float* point,const std::vector<int>& vectorIndices);
    
        PCNode* buildNode(std::vector<int> &vectorIndices,const std::vector<float>& parentPC);

    
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
        PCTree(const char* data_level0_memory, size_t data_size, size_t dim,size_t num_vectors, int leafCapacity);
        ~PCTree();
        std::vector<int> searchNN(const float* query) const;
        int getHeight(PCNode* root) const;
        int getHeight() const;
};

