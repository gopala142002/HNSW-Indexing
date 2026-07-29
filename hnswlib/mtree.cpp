#include "mtree.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

MTNode::~MTNode() 
{
    for (RoutingEntry& entry : routingEntries) 
    {
        delete entry.child;
    }
}

MTree::MTree(char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int maxRoutingEntries,int maxLeafObjects)
{
    this->data_level0_memory=data_level0_memory;
    this->data_size=data_size;
    this->dim=dim;
    this->num_vectors=num_vectors;
    this->maxRoutingEntries=maxRoutingEntries;
    this->maxLeafObjects=maxLeafObjects;
}

MTree::MTree(char* data_level0_memory,size_t data_size,size_t vector_offset,size_t dim,size_t num_vectors,int maxRoutingEntries,int maxLeafObjects)
{
    this->data_level0_memory=data_level0_memory;
    this->data_size=data_size;
    this->vector_offset=vector_offset;
    this->dim=dim;
    this->num_vectors=num_vectors;
    this->maxRoutingEntries=maxRoutingEntries;
    this->maxLeafObjects=maxLeafObjects;   
    
    if (this->data_level0_memory == nullptr) 
    {
        throw std::invalid_argument("MTree: data_level0_memory must not be null");
    }
    if (this->maxRoutingEntries < 2) 
    {
        this->maxRoutingEntries= 2;
    }
    if (this->maxLeafObjects < 2) 
    {
        this->maxLeafObjects= 2;
    } 
}


MTree::~MTree() 
{
    delete root;
}

const float* MTree::getVector(int id) const 
{
    if (id < 0 || static_cast<size_t>(id) >= num_vectors) 
    {
        return nullptr;
    }
    return reinterpret_cast<const float*>(data_level0_memory + static_cast<size_t>(id) * data_size + vector_offset);
}

float MTree::distance(const float* a, const float* b) const 
{
    float sumSq = 0.0f;
    for (size_t i = 0; i < dim; ++i) 
    {
        const float diff = a[i] - b[i];
        sumSq += diff * diff;
    }
    return sumSq;
}

float MTree::distanceToId(int a, int b) const 
{
    const float* va = getVector(a);
    const float* vb = getVector(b);
    if (va == nullptr || vb == nullptr) {
        return std::numeric_limits<float>::infinity();
    }
    return distance(va, vb);
}

void MTree::build() {
    delete root;
    root = nullptr;

    if (num_vectors == 0) 
    {
        root = new MTNode();
        root->isLeaf = true;
        return;
    }

    root = new MTNode();
    root->isLeaf = true;
    for (size_t i = 0; i < num_vectors; ++i) 
    {
        insertObject(static_cast<int>(i));
    }
}

void MTree::insert(int vectorID) 
{
    if (vectorID < 0) 
    {
        return;
    }
    if (static_cast<size_t>(vectorID) >= num_vectors) 
    {
        num_vectors = static_cast<size_t>(vectorID) + 1;
    }
    insertObject(vectorID);
}

void MTree::insertObject(int vectorID) 
{
    if (root == nullptr) {
        root = new MTNode();
        root->isLeaf = true;
    }

    InsertResult result = insertIntoNode(root, vectorID);
    if (result.split) {
        MTNode* newRoot = new MTNode();
        newRoot->isLeaf = false;
        newRoot->routingEntries = result.entries;
        root = newRoot;
    }
}

MTree::InsertResult MTree::insertIntoNode(MTNode* node, int vectorID) 
{
    InsertResult result;
    result.split = false;

    if (node == nullptr) 
    {
        return result;
    }

    if (node->isLeaf) 
    {
        ObjectEntry entry;
        entry.vectorID = vectorID;
        node->objectEntries.push_back(entry);
        if (node->objectEntries.size() > static_cast<size_t>(maxLeafObjects)) 
        {
            result = splitNode(node);
        }
        return result;
    }

    int bestEntry = -1;
    float bestDist = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < node->routingEntries.size(); ++i) 
    {
        RoutingEntry& entry = node->routingEntries[i];
        const float d = distanceToId(vectorID, entry.pivotID);
        if (d < bestDist) 
        {
            bestDist = d;
            bestEntry = static_cast<int>(i);
        }
    }

    if (bestEntry >= 0) 
    {
        result = insertIntoNode(node->routingEntries[bestEntry].child, vectorID);
        if (result.split) 
        {
            node->routingEntries.erase(node->routingEntries.begin() + bestEntry);
            node->routingEntries.insert(node->routingEntries.end(), result.entries.begin(), result.entries.end());
            if (node->routingEntries.size() > static_cast<size_t>(maxRoutingEntries)) 
            {
                result = splitNode(node);
            } 
            else 
            {
                result.split = false;
            }
        }
    }
    return result;
}

MTree::InsertResult MTree::splitNode(MTNode* node) 
{
    InsertResult result;
    result.split = true;

    if (node == nullptr) 
    {
        return result;
    }

    if (node->isLeaf) 
    {
        std::vector<int> ids;
        ids.reserve(node->objectEntries.size());
        for (const ObjectEntry& entry : node->objectEntries) 
        {
            ids.push_back(entry.vectorID);
        }
        if (ids.size() < 2) 
        {
            return result;
        }

        std::pair<int, int> promoters = selectPromoters(ids);
        int first = promoters.first;
        int second = promoters.second;

        MTNode* left = node;
        MTNode* right = new MTNode();
        right->isLeaf = true;

        left->objectEntries.clear();
        right->objectEntries.clear();

        for (int id : ids) 
        {
            const float d1 = distanceToId(id, first);
            const float d2 = distanceToId(id, second);
            ObjectEntry obj;
            obj.vectorID = id;
            if (d1 <= d2) 
            {
                left->objectEntries.push_back(obj);
            } 
            else 
            {
                right->objectEntries.push_back(obj);
            }
        }

        if (left->objectEntries.empty()) 
        {
            left->objectEntries.push_back({first});
        }
        if (right->objectEntries.empty()) 
        {
            right->objectEntries.push_back({second});
        }

        RoutingEntry leftEntry;
        leftEntry.pivotID = first;
        leftEntry.child = left;

        RoutingEntry rightEntry;
        rightEntry.pivotID = second;
        rightEntry.child = right;

        result.entries.push_back(leftEntry);
        result.entries.push_back(rightEntry);
        return result;
    }

    std::vector<int> ids;
    ids.reserve(node->routingEntries.size());
    for (const RoutingEntry& entry : node->routingEntries) 
    {
        ids.push_back(entry.pivotID);
    }

    if (ids.size() < 2) 
    {
        return result;
    }

    std::pair<int, int> promoters = selectPromoters(ids);
    int first = promoters.first;
    int second = promoters.second;

    MTNode* left = node;
    MTNode* right = new MTNode();
    right->isLeaf = false;

    left->routingEntries.clear();
    left->objectEntries.clear();
    right->routingEntries.clear();
    right->objectEntries.clear();

    for (const RoutingEntry& entry : node->routingEntries) 
    {
        const float d1 = distanceToId(entry.pivotID, first);
        const float d2 = distanceToId(entry.pivotID, second);
        RoutingEntry copy = entry;
        if (d1 <= d2) 
        {
            left->routingEntries.push_back(copy);
        } 
        else 
        {
            right->routingEntries.push_back(copy);
        }
    }

    RoutingEntry leftEntry;
    leftEntry.pivotID = first;
    leftEntry.child = left;

    RoutingEntry rightEntry;
    rightEntry.pivotID = second;
    rightEntry.child = right;

    result.entries.push_back(leftEntry);
    result.entries.push_back(rightEntry);
    return result;
}

std::pair<int, int> MTree::selectPromoters(const std::vector<int>& ids) const 
{
    if (ids.size() < 2) 
    {
        return {ids.empty() ? -1 : ids[0], -1};
    }

    int first = ids[0];
    int second = ids[0];
    float best = -1.0f;
    for (size_t i = 0; i < ids.size(); ++i) 
    {
        for (size_t j = i + 1; j < ids.size(); ++j) 
        {
            const float d = distanceToId(ids[i], ids[j]);
            if (d > best) 
            {
                best = d;
                first = ids[i];
                second = ids[j];
            }
        }
    }
    return {first, second};
}

int MTree::greedySearch(const float* query) const 
{
    if (root == nullptr || query == nullptr) 
    {
        return -1;
    }

    const MTNode* node = root;
    while (!node->isLeaf) 
    {
        if (node->routingEntries.empty()) 
        {
            break;
        }

        int bestEntry = -1;
        float bestDist = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < node->routingEntries.size(); ++i) 
        {
            const RoutingEntry& entry = node->routingEntries[i];
            const float d = distance(query, getVector(entry.pivotID));
            if (d < bestDist) 
            {
                bestDist = d;
                bestEntry = static_cast<int>(i);
            }
        }
        if (bestEntry < 0) 
        {
            break;
        }
        node = node->routingEntries[bestEntry].child;
    }

    if (node == nullptr || node->objectEntries.empty()) 
    {
        return -1;
    }

    int bestId = -1;
    float bestDist = std::numeric_limits<float>::infinity();
    for (const ObjectEntry& entry : node->objectEntries) 
    {
        const float d = distance(query, getVector(entry.vectorID));
        if (d < bestDist) 
        {
            bestDist = d;
            bestId = entry.vectorID;
        }
    }

    return bestId;
}

int MTree::getHeight() const
{
    return calculateHeight(root);
}

int MTree::calculateHeight(const MTNode* node) const
{
    if (node == nullptr || node->isLeaf)
    {
        return 0;
    }

    int maxChildHeight = 0;
    for (const RoutingEntry& entry : node->routingEntries)
    {
        maxChildHeight = std::max(maxChildHeight, calculateHeight(entry.child));
    }
    return 1 + maxChildHeight;
}
