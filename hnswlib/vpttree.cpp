#include "vpttree.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>


VPTNode::~VPTNode() 
{
    delete left;
    delete right;
}

VantagePointTree::VantagePointTree(char* data_level0_memory, size_t data_size, size_t dim, size_t num_vectors, int leafCapacity, int leafClusters)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->leafCapacity = leafCapacity;
    this->leafClusters = leafClusters < 1 ? 1 : leafClusters;

    if (data_level0_memory == nullptr) 
    {
        throw std::invalid_argument("VantagePointTree: data_level0_memory must not be null");
    }
    if (leafCapacity < 1) 
    {
        throw std::invalid_argument("VantagePointTree: leafCapacity must be >= 1");
    }
}

VantagePointTree::~VantagePointTree() 
{
    delete root;
}

const float* VantagePointTree::getVector(int id) const 
{
    return reinterpret_cast<const float*>(data_level0_memory + static_cast<size_t>(id) * data_size);
}

float VantagePointTree::distance(const float* a, const float* b) const 
{
    float sumSq = 0.0f;
    for (size_t i = 0; i < dim; ++i) 
    {
        const float diff = a[i] - b[i];
        sumSq += diff * diff;
    }
    return sumSq;
}

int VantagePointTree::choosePivot(const std::vector<int>& indices) const 
{
    if (indices.empty())
        return -1;
    
    // Simple strategy: choose a random point as pivot
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> uniform(0, static_cast<int>(indices.size()) - 1);
    
    int pivotIdx = uniform(rng);
    return indices[pivotIdx];
}

void VantagePointTree::build() 
{
    delete root;
    root = nullptr;

    if (num_vectors == 0) 
    {
        root = new VPTNode();
        root->isLeaf = true;
        return;
    }

    std::vector<int> allIndices(num_vectors);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    root = buildRecursive(std::move(allIndices));
    preprocessLeafEntryPoints(root);
}

VPTNode* VantagePointTree::buildRecursive(std::vector<int>&& indices) 
{
    VPTNode* node = new VPTNode();

    const int n = static_cast<int>(indices.size());

    // Base case: create leaf node
    if (n <= leafCapacity) 
    {
        node->isLeaf = true;
        node->vectorIndices = std::move(indices);
        return node;
    }

    // Choose a pivot point from the indices
    int pivotId = choosePivot(indices);
    if (pivotId < 0) 
    {
        node->isLeaf = true;
        node->vectorIndices = std::move(indices);
        return node;
    }

    node->pivot = pivotId;
    node->isLeaf = false;

    // Calculate distances from pivot to all points
    const float* pivotVec = getVector(pivotId);
    std::vector<float> distances;
    distances.reserve(indices.size());

    for (int idx : indices) 
    {
        float d = distance(pivotVec, getVector(idx));
        distances.push_back(d);
    }

    // Find median distance
    std::vector<float> sortedDist = distances;
    std::nth_element(sortedDist.begin(), sortedDist.begin() + sortedDist.size() / 2, sortedDist.end());
    node->median_distance = sortedDist[sortedDist.size() / 2];

    // Partition indices based on median distance
    std::vector<int> leftIndices, rightIndices;
    leftIndices.reserve(indices.size());
    rightIndices.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) 
    {
        if (distances[i] <= node->median_distance) 
        {
            leftIndices.push_back(indices[i]);
        } 
        else 
        {
            rightIndices.push_back(indices[i]);
        }
    }

    // Ensure both partitions are non-empty
    if (leftIndices.empty()) 
    {
        leftIndices.push_back(rightIndices.back());
        rightIndices.pop_back();
    }
    if (rightIndices.empty()) 
    {
        rightIndices.push_back(leftIndices.back());
        leftIndices.pop_back();
    }

    // Recursively build left and right subtrees
    node->left = buildRecursive(std::move(leftIndices));
    node->right = buildRecursive(std::move(rightIndices));

    if (node->left)
        node->left->parent = node;
    if (node->right)
        node->right->parent = node;

    return node;
}


std::vector<float> VantagePointTree::computeLeafCentroid(const VPTNode* leaf) const
{
    std::vector<float> centroid(dim, 0.0f);
    if (leaf == nullptr || leaf->vectorIndices.empty())
    {
        return centroid;
    }
    for (int id : leaf->vectorIndices)
    {
        const float* vec = getVector(id);
        for (size_t d = 0; d < dim; ++d)
        {
            centroid[d] += vec[d];
        }
    }
    const float invCount =1.0f / static_cast<float>(leaf->vectorIndices.size());
    for (size_t d = 0; d < dim; ++d)
    {
        centroid[d] *= invCount;
    }
    return centroid;
}
int VantagePointTree::findNearestToCentroid(const VPTNode* leaf,const std::vector<float>& centroid) const
{
    if (leaf == nullptr || leaf->vectorIndices.empty())
    {
        return -1;
    }
    int nearestId = -1;
    float minDist = std::numeric_limits<float>::max();
    for (int id : leaf->vectorIndices)
    {
        const float d =distance(centroid.data(), getVector(id));
        if (d < minDist)
        {
            minDist = d;
            nearestId = id;
        }
    }
    return nearestId;
}


int VantagePointTree::findNearestToCentroidVec(const std::vector<float>& centroid,const std::vector<int>& vectorIndices) const
{
    if (vectorIndices.empty())
    {
        return -1;
    }
    int nearestId = vectorIndices[0];
    float minDist = distance(centroid.data(), getVector(nearestId));
    for (int id : vectorIndices)
    {
        const float d = distance(centroid.data(), getVector(id));
        if (d < minDist)
        {
            minDist = d;
            nearestId = id;
        }
    }
    return nearestId;
}


int VantagePointTree::findNearestCentroidIdx(const float* vector,const std::vector<std::vector<float>>& centroids) const
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


void VantagePointTree::initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const
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


void VantagePointTree::runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const
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


void VantagePointTree::populateLeafMulti(VPTNode* node)
{
    if (node == nullptr || node->vectorIndices.empty())
    {
        return;
    }

    // leafClusters == 1 naturally reduces to a single cluster covering
    // the whole leaf (still independent of centroidEntryPoint above).
    std::vector<std::vector<float>> leafCentroids;
    std::vector<std::vector<int>> leafClusterAssignments;

    initializeLeafCentroids(node->vectorIndices, this->leafClusters, leafCentroids);
    runLeafKMeans(node->vectorIndices, leafCentroids, leafClusterAssignments);

    node->leafClusterCentroids = std::move(leafCentroids);

    node->leafClusterRepresentatives.clear();
    node->leafClusterRepresentatives.reserve(node->leafClusterCentroids.size());
    for (size_t c = 0; c < node->leafClusterCentroids.size(); ++c)
    {
        int representative = findNearestToCentroidVec(node->leafClusterCentroids[c], leafClusterAssignments[c]);
        node->leafClusterRepresentatives.push_back(representative);
    }
}


void VantagePointTree::preprocessLeafEntryPoints(VPTNode* node)
{
    if (node == nullptr)
    {
        return;
    }
    if (node->isLeaf)
    {
        if (!node->vectorIndices.empty())
        {
            std::vector<float> centroid =
                computeLeafCentroid(node);

            node->centroidEntryPoint =
                findNearestToCentroid(node, centroid);

            // Approach 2: leaf-level K-Means for multiple entry points
            populateLeafMulti(node);
        }

        return;
    }
    preprocessLeafEntryPoints(node->left);
    preprocessLeafEntryPoints(node->right);
}

int VantagePointTree::searchEntryPoint(const float* query) const
{
    const VPTNode* node = findLeafNode(query);
    if (node == nullptr)
    {
        return -1;
    }
    return node->centroidEntryPoint;
}


const VPTNode* VantagePointTree::findLeafNode(const float* query) const
{
    if (root == nullptr || query == nullptr)
    {
        return nullptr;
    }
    const VPTNode* current = root;
    while (current != nullptr && !current->isLeaf)
    {
        const float d = distance(query, getVector(current->pivot));
        if (d <= current->median_distance)
        {
            current = current->left;
        }
        else
        {
            current = current->right;
        }
    }
    return current;
}


std::vector<int> VantagePointTree::searchEntryPointMulti(const float* query) const
{
    const VPTNode* node = findLeafNode(query);
    if (node == nullptr)
    {
        return {};
    }
    return node->leafClusterRepresentatives;
}

std::vector<int> VantagePointTree::searchNN(const float* query) const 
{
    std::vector<int> collected;

    if (root == nullptr || query == nullptr) 
    {
        return collected;
    }

    // Greedy search: traverse the tree following the path closest to query
    VPTNode* current = root;

    while (current != nullptr) 
    {
        if (current->isLeaf) 
        {
            // Collect all points from the leaf
            collected.insert(collected.end(), current->vectorIndices.begin(), current->vectorIndices.end());
            break;
        }

        // Calculate distance from query to pivot
        const float d = distance(query, getVector(current->pivot));
        
        // Also collect the pivot itself as a candidate
        collected.push_back(current->pivot);

        // Decide which subtree to traverse
        if (d <= current->median_distance) 
        {
            current = current->left;
        } 
        else 
        {
            current = current->right;
        }
    }

    return collected;
}

int VantagePointTree::getHeight() const
{
    return calculateHeight(root);
}

int VantagePointTree::calculateHeight(VPTNode* node) const
{
    if (node == nullptr || node->isLeaf)
    {
        return 0;
    }

    int leftHeight = calculateHeight(node->left);
    int rightHeight = calculateHeight(node->right);

    return 1 + std::max(leftHeight, rightHeight);
}