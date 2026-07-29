#pragma once
#define MTREE_H

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

    MTNode() = default;
    MTNode(const MTNode&) = delete;
    MTNode& operator=(const MTNode&) = delete;
    ~MTNode();
};

class MTree 
{
    public:
        MTree(char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int maxRoutingEntries = 8,int maxLeafObjects = 32);

        MTree(char* data_level0_memory,size_t data_size,size_t vector_offset,size_t dim,size_t num_vectors,int maxRoutingEntries,int maxLeafObjects);

        ~MTree();

        MTree(const MTree&) = delete;
        MTree& operator=(const MTree&) = delete;

        void build();
        void insert(int vectorID);

        int greedySearch(const float* query) const;
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
};


