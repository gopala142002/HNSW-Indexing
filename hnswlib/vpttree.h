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

    VPTNode() = default;
    VPTNode(const VPTNode&) = delete;
    VPTNode& operator=(const VPTNode&) = delete;

    ~VPTNode();
};

class VantagePointTree
{
    public:
    VantagePointTree(char* data_level0_memory, size_t data_size, size_t dim, size_t num_vectors, int leafCapacity);

    ~VantagePointTree();

    VantagePointTree(const VantagePointTree&) = delete;
    VantagePointTree& operator=(const VantagePointTree&) = delete;

    void build();

    std::vector<int> searchNN(const float* query) const;
    int getHeight() const;

    VPTNode* root = nullptr;

    private:
    char* data_level0_memory;
    size_t data_size;
    size_t dim;
    size_t num_vectors;
    int leafCapacity;

    const float* getVector(int id) const;

    float distance(const float* a, const float* b) const;

    VPTNode* buildRecursive(std::vector<int>&& indices);
    int calculateHeight(VPTNode* node) const;

    int choosePivot(const std::vector<int>& indices) const;
};
