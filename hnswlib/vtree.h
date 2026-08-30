#pragma once
#define VTREE_H
#include <vector>
struct VTNode;

struct Branch
{
    int pivot;
    float radius;
    VTNode* child;
};

struct VTNode
{
    bool isLeaf = false;
    std::vector<int> vectorIndices;
    std::vector<Branch> branches;
    VTNode* parent = nullptr;
    int centroidEntryPoint = -1;
    VTNode() = default;
    VTNode(const VTNode&) = delete;
    VTNode& operator=(const VTNode&) = delete;

    ~VTNode();
};

class VoronoiTree
{
    public:
    VoronoiTree(char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int numPivots,int leafCapacity);

    ~VoronoiTree();

    VoronoiTree(const VoronoiTree&) = delete;
    VoronoiTree& operator=(const VoronoiTree&) = delete;

    void build();

    std::vector<int> searchNN(const float* query) const;
    int searchEntryPoint(const float* query) const;
    int getHeight() const;

    VTNode* root = nullptr;

    private:
    char* data_level0_memory;
    size_t data_size;
    size_t dim;
    size_t num_vectors;

    int numPivots;
    int leafCapacity;

    const float* getVector(int id) const;

    float distance(const float* a, const float* b) const;

    VTNode* buildRecursive(std::vector<int>&& indices);
    int calculateHeight(VTNode* node) const;

    void choosePivots(const std::vector<int>& indices,std::vector<int>& pivots,std::vector<float>& distanceMatrix) const;
    void partition(const std::vector<int>& indices,const std::vector<int>& pivots,const std::vector<float>& distanceMatrix,std::vector<std::vector<int>>& clusters,std::vector<float>& radii) const;

    std::vector<float> computeLeafCentroid(const VTNode* leaf) const;
    int findNearestToCentroid(const VTNode* leaf,const std::vector<float>& centroid) const;
    void preprocessLeafEntryPoints(VTNode* node);
};

