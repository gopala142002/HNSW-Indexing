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

MTree::MTree(char* data_level0_memory, size_t data_size, size_t dim, size_t num_vectors, int maxRoutingEntries, int maxLeafObjects, int leafClusters)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->vector_offset = 0;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->maxRoutingEntries = maxRoutingEntries;
    this->maxLeafObjects = maxLeafObjects;
    this->leafClusters = leafClusters < 1 ? 1 : leafClusters;
}

MTree::MTree(char* data_level0_memory, size_t data_size, size_t vector_offset, size_t dim, size_t num_vectors, int maxRoutingEntries, int maxLeafObjects, int leafClusters)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->vector_offset = vector_offset;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->maxRoutingEntries = maxRoutingEntries;
    this->maxLeafObjects = maxLeafObjects;   
    this->leafClusters = leafClusters;
    
    if (this->data_level0_memory == nullptr) 
    {
        throw std::invalid_argument("MTree: data_level0_memory must not be null");
    }
    if (this->maxRoutingEntries < 2) 
    {
        this->maxRoutingEntries = 2;
    }
    if (this->maxLeafObjects < 2) 
    {
        this->maxLeafObjects = 2;
    } 
    if (this->leafClusters < 1) 
    {
        this->leafClusters = 1;
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
    preprocessLeafEntryPoints(root);
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
            // Remove the child entry that split and insert the two new split child entries
            node->routingEntries.erase(node->routingEntries.begin() + bestEntry);
            node->routingEntries.insert(node->routingEntries.end(), result.entries.begin(), result.entries.end());
            
            // If this internal node exceeds maximum capacity, split it and return the result upwards
            if (node->routingEntries.size() > static_cast<size_t>(maxRoutingEntries)) 
            {
                return splitNode(node);
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

    // --- LEAF NODE SPLIT ---
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

    // --- INTERNAL NODE SPLIT (FIXED) ---

    // 1. Preserve child subtrees in oldEntries before clearing
    std::vector<RoutingEntry> oldEntries = node->routingEntries;

    std::vector<int> ids;
    ids.reserve(oldEntries.size());
    for (const RoutingEntry& entry : oldEntries) 
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
    left->isLeaf = false;
    right->isLeaf = false;

    // 2. Clear old routing lists
    left->routingEntries.clear();
    left->objectEntries.clear();
    right->routingEntries.clear();
    right->objectEntries.clear();

    // 3. Redistribute child branches from saved oldEntries
    for (const RoutingEntry& entry : oldEntries) 
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

    // Ensure neither left nor right internal node is left completely empty
    if (left->routingEntries.empty() && !right->routingEntries.empty()) 
    {
        left->routingEntries.push_back(right->routingEntries.back());
        right->routingEntries.pop_back();
    } 
    else if (right->routingEntries.empty() && !left->routingEntries.empty()) 
    {
        right->routingEntries.push_back(left->routingEntries.back());
        left->routingEntries.pop_back();
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


const MTNode* MTree::findLeafNode(const float* query) const
{
    if (root == nullptr || query == nullptr)
    {
        return nullptr;
    }
    const MTNode* node = root;
    while (node != nullptr && !node->isLeaf)
    {
        if (node->routingEntries.empty())
        {
            return nullptr;
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
            return nullptr;
        }
        node = node->routingEntries[bestEntry].child;
    }
    return node;
}


int MTree::searchEntryPoint(const float* query) const
{
    const MTNode* node = findLeafNode(query);
    if (node == nullptr)
    {
        return -1;
    }
    return node->centroidEntryPoint;
}


std::vector<int> MTree::searchEntryPointMulti(const float* query) const
{
    const MTNode* node = findLeafNode(query);
    if (node == nullptr)
    {
        return {};
    }
    return node->leafClusterRepresentatives;
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

std::vector<float> MTree::computeLeafCentroid(const MTNode* leaf) const
{
    std::vector<float> centroid(dim, 0.0f);
    if (leaf == nullptr || leaf->objectEntries.empty())
    {
        return centroid;
    }
    for (const ObjectEntry& entry : leaf->objectEntries)
    {
        const float* vec = getVector(entry.vectorID);
        if (vec == nullptr)
        {
            continue;
        }
        for (size_t d = 0; d < dim; ++d)
        {
            centroid[d] += vec[d];
        }
    }
    float invCount =1.0f / static_cast<float>(leaf->objectEntries.size());
    for (size_t d = 0; d < dim; ++d)
    {
        centroid[d] *= invCount;
    }
    return centroid;
}
int MTree::findNearestToCentroid(const MTNode* leaf,const std::vector<float>& centroid) const
{
    if (leaf == nullptr || leaf->objectEntries.empty())
    {
        return -1;
    }
    int nearestID = -1;
    float minDist = std::numeric_limits<float>::infinity();
    for (const ObjectEntry& entry : leaf->objectEntries)
    {
        const float* vec = getVector(entry.vectorID);
        if (vec == nullptr)
        {
            continue;
        }
        float dist = distance(centroid.data(),vec);
        if (dist < minDist)
        {
            minDist = dist;
            nearestID = entry.vectorID;
        }
    }
    return nearestID;
}


int MTree::findNearestToCentroidVec(const std::vector<float>& centroid,const std::vector<int>& vectorIndices) const
{
    if (vectorIndices.empty())
        return -1;
    int nearestID = -1;
    float minDist = std::numeric_limits<float>::infinity();

    for (int vectorID : vectorIndices)
    {
        const float* vec = getVector(vectorID);
        if (vec == nullptr)
            continue;
        float dist = distance(centroid.data(), vec);
        if (dist < minDist)
        {
            minDist = dist;
            nearestID = vectorID;
        }
    }
    return nearestID;
}

int MTree::findNearestCentroidIdx(const float* vector,const std::vector<std::vector<float>>& centroids) const
{
    int nearest = 0;
    float minDistance = distance(vector, centroids[0].data());
    for (size_t c = 1; c < centroids.size(); ++c)
    {
        float d = distance(vector, centroids[c].data());
        if (d < minDistance)
        {
            minDistance = d;
            nearest = static_cast<int>(c);
        }
    }
    return nearest;
}


void MTree::initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const
{
    const int n = static_cast<int>(vectorIndices.size());
    k = std::min(k, n);
    centroids.clear();
    centroids.resize(k, std::vector<float>(dim, 0.0f));

    for (int c = 0; c < k; ++c)
    {
        int position = (static_cast<long long>(c) * n) / k;
        if (position >= n)
        {
            position = n - 1;
        }
        const float* vec = getVector(vectorIndices[position]);
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[c][d] = vec[d];
        }
    }
}


void MTree::runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const
{
    const int k = static_cast<int>(centroids.size());
    const int n = static_cast<int>(vectorIndices.size());
    if (k == 0 || n == 0)
    {
        clusters.clear();
        return;
    }

    std::vector<int> assignments(n, -1);

    for (int iteration = 0; iteration < MAX_KMEANS_ITERATIONS; ++iteration)
    {
        clusters.assign(k, std::vector<int>());
        for (int i = 0; i < n; ++i)
        {
            int vectorID = vectorIndices[i];
            int clusterID = findNearestCentroidIdx(getVector(vectorID), centroids);
            assignments[i] = clusterID;
            clusters[clusterID].push_back(vectorID);
        }

        std::vector<std::vector<float>> newCentroids(k, std::vector<float>(dim, 0.0f));
        std::vector<int> clusterCounts(k, 0);
        for (int i = 0; i < n; ++i)
        {
            int clusterID = assignments[i];
            const float* vec = getVector(vectorIndices[i]);
            clusterCounts[clusterID]++;
            for (size_t d = 0; d < dim; ++d)
            {
                newCentroids[clusterID][d] += vec[d];
            }
        }

        for (int c = 0; c < k; ++c)
        {
            if (clusterCounts[c] == 0)
            {
                int farthestIndex = -1;
                float farthestDistance = -1.0f;
                for (int i = 0; i < n; ++i)
                {
                    int assigned = assignments[i];
                    float d = distance(getVector(vectorIndices[i]), centroids[assigned].data());
                    if (d > farthestDistance)
                    {
                        farthestDistance = d;
                        farthestIndex = i;
                    }
                }
                if (farthestIndex >= 0)
                {
                    const float* vec = getVector(vectorIndices[farthestIndex]);
                    for (size_t d = 0; d < dim; ++d)
                    {
                        newCentroids[c][d] = vec[d];
                    }
                    clusterCounts[c] = 1;
                }
            }
        }

        for (int c = 0; c < k; ++c)
        {
            if (clusterCounts[c] > 0)
            {
                float invCount = 1.0f / static_cast<float>(clusterCounts[c]);
                for (size_t d = 0; d < dim; ++d)
                {
                    newCentroids[c][d] *= invCount;
                }
            }
        }

        float maxMovement = 0.0f;
        for (int c = 0; c < k; ++c)
        {
            float movement = 0.0f;
            for (size_t d = 0; d < dim; ++d)
            {
                float diff = newCentroids[c][d] - centroids[c][d];
                movement += diff * diff;
            }
            maxMovement = std::max(maxMovement, movement);
        }

        centroids = std::move(newCentroids);
        if (maxMovement < KMEANS_TOLERANCE * KMEANS_TOLERANCE)
        {
            break;
        }
    }

    clusters.assign(k, std::vector<int>());
    for (int vectorID : vectorIndices)
    {
        int clusterID = findNearestCentroidIdx(getVector(vectorID), centroids);
        clusters[clusterID].push_back(vectorID);
    }
}


void MTree::populateLeafMulti(MTNode* node)
{
    if (node == nullptr || node->objectEntries.empty())
    {
        return;
    }

    std::vector<int> ids;
    ids.reserve(node->objectEntries.size());
    for (const ObjectEntry& entry : node->objectEntries)
    {
        ids.push_back(entry.vectorID);
    }

    // leafClusters == 1 naturally reduces to a single cluster covering
    // the whole leaf (still independent of centroidEntryPoint above).
    std::vector<std::vector<float>> leafCentroids;
    std::vector<std::vector<int>> leafClusterAssignments;

    initializeLeafCentroids(ids, this->leafClusters, leafCentroids);
    runLeafKMeans(ids, leafCentroids, leafClusterAssignments);

    node->leafClusterCentroids = std::move(leafCentroids);

    node->leafClusterRepresentatives.clear();
    node->leafClusterRepresentatives.reserve(node->leafClusterCentroids.size());
    for (size_t c = 0; c < node->leafClusterCentroids.size(); ++c)
    {
        int representative = findNearestToCentroidVec(node->leafClusterCentroids[c], leafClusterAssignments[c]);
        node->leafClusterRepresentatives.push_back(representative);
    }
}


void MTree::preprocessLeafEntryPoints(MTNode* node)
{
    if (node == nullptr)
    {
        return;
    }
    // Leaf node
    if (node->isLeaf)
    {
        if (!node->objectEntries.empty())
        {
            std::vector<float> centroid =
                computeLeafCentroid(node);
            node->centroidEntryPoint =findNearestToCentroid(node,centroid);

            // Approach 2: leaf-level K-Means for multiple entry points
            populateLeafMulti(node);
        }
        return;
    }
    // Internal node: recursively process all children.
    for (RoutingEntry& entry : node->routingEntries)
    {
        preprocessLeafEntryPoints(entry.child);
    }
}