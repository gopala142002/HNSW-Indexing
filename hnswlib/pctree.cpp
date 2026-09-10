#include "pctree.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>


PCTree::PCTree(const char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int leafCapacity,int numPartitions,int leafClusters)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->leafCapacity = leafCapacity;
    this->numPartitions = numPartitions;
    this->leafClusters = leafClusters;

    if (data_level0_memory == nullptr)
    {
        throw std::invalid_argument("PCTree: data_level0_memory must not be null");
    }
    if (leafCapacity < 1)
    {
        throw std::invalid_argument("PCTree: leafCapacity must be >= 1");
    }
    if (numPartitions < 2)
    {
        throw std::invalid_argument("PCTree: numPartitions must be >= 2");
    }
    if (leafClusters < 1)
    {
        throw std::invalid_argument("PCTree: leafClusters must be >= 1");
    }
    std::vector<int> allIndices(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i)
    {
        allIndices[i] = static_cast<int>(i);
    }
    std::vector<float> initialPC(dim, 0.0f);

    if (dim > 0)
    {
        initialPC[0] = 1.0f;
    }
    root = buildNode(allIndices, initialPC);
}


PCTree::~PCTree()
{
    delete root;
}


std::vector<float> PCTree::computeCentroid(const std::vector<int>& vectorIndices)
{
    std::vector<float> centroid(dim, 0.0f);
    if (vectorIndices.empty())
    {
        return centroid;
    }
    for (int idx : vectorIndices)
    {
        const float* vec = getVector(idx);
        for (size_t d = 0; d < dim; ++d)
        {
            centroid[d] += vec[d];
        }
    }
    float invCount = 1.0f / vectorIndices.size();
    for (size_t d = 0; d < dim; ++d)
    {
        centroid[d] *= invCount;
    }
    return centroid;
}


void PCTree::computePCA(const std::vector<int>& vectorIndices,std::vector<float>& pc)
{
    std::vector<float> centroid =computeCentroid(vectorIndices);

    if (pc.empty() || pc[0] == 0.0f)
    {
        pc.assign(dim,1.0f / std::sqrt(static_cast<float>(dim)));
    }

    const int PC_itr = 20;
    const float acceptable_error = 1e-6f;

    std::vector<float> next_pc(dim);
    std::vector<float> centered(dim);

    for (int iter = 0; iter < PC_itr; ++iter)
    {
        std::fill(next_pc.begin(),next_pc.end(),0.0f);

        for (int idx : vectorIndices)
        {
            const float* vec = getVector(idx);
            for (size_t d = 0; d < dim; ++d)
            {
                centered[d] =vec[d] - centroid[d];
            }
            float projection = 0.0f;
            for (size_t d = 0; d < dim; ++d)
            {
                projection +=centered[d] * pc[d];
            }
            for (size_t d = 0; d < dim; ++d)
            {
                next_pc[d] +=projection * centered[d];
            }
        }
        float norm = 0.0f;
        for (size_t d = 0; d < dim; ++d)
        {
            norm +=next_pc[d] * next_pc[d];
        }

        if (norm < 1e-10f)
        {
            break;
        }

        norm = std::sqrt(norm);

        float dot_diff = 0.0f;

        for (size_t d = 0; d < dim; ++d)
        {
            float normalized =next_pc[d] / norm;
            dot_diff +=std::abs(normalized - pc[d]);
            pc[d] = normalized;
        }

        if (dot_diff < acceptable_error)
        {
            break;
        }
    }
}


int PCTree::findNearestToPoint(const float* point,const std::vector<int>& vectorIndices)
{
    if (vectorIndices.empty())
    {
        return -1;
    }
    int nearest = vectorIndices[0];

    float minDist =squaredL2Distance(point,getVector(nearest));

    for (int idx : vectorIndices)
    {
        float dist =squaredL2Distance(point,getVector(idx));
        if (dist < minDist)
        {
            minDist = dist;
            nearest = idx;
        }
    }
    return nearest;
}


int PCTree::findNearestCentroid(const float* vector,const std::vector<std::vector<float>>& centroids) const
{
    int nearest = 0;
    float minDistance = squaredL2Distance(vector, centroids[0].data());
    for (size_t c = 1; c < centroids.size(); ++c)
    {
        float distance = squaredL2Distance(vector, centroids[c].data());
        if (distance < minDistance)
        {
            minDistance = distance;
            nearest = static_cast<int>(c);
        }
    }
    return nearest;
}


void PCTree::initializeLeafCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const
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


void PCTree::runLeafKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const
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
            int clusterID = findNearestCentroid(getVector(vectorID), centroids);
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
                    float distance = squaredL2Distance(getVector(vectorIndices[i]), centroids[assigned].data());
                    if (distance > farthestDistance)
                    {
                        farthestDistance = distance;
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
        int clusterID = findNearestCentroid(getVector(vectorID), centroids);
        clusters[clusterID].push_back(vectorID);
    }
}


void PCTree::populateLeaf(PCNode* node)
{
    // --- Approach 1: single leaf representative (unchanged behavior) ---
    std::vector<float> centroid = computeCentroid(node->vectorIndices);
    node->representative = findNearestToPoint(centroid.data(), node->vectorIndices);

    // --- Approach 2: leaf-level K-Means for multiple HNSW entry points ---
    // leafClusters == 1 naturally reduces to the same "nearest vector to
    // leaf centroid" result as Approach 1 above.
    std::vector<std::vector<float>> leafCentroids;
    std::vector<std::vector<int>> leafClusterAssignments;

    initializeLeafCentroids(node->vectorIndices, this->leafClusters, leafCentroids);
    runLeafKMeans(node->vectorIndices, leafCentroids, leafClusterAssignments);

    node->leafClusterCentroids = std::move(leafCentroids);

    node->leafClusterRepresentatives.clear();
    node->leafClusterRepresentatives.reserve(node->leafClusterCentroids.size());
    for (size_t c = 0; c < node->leafClusterCentroids.size(); ++c)
    {
        int representative = findNearestToPoint(
            node->leafClusterCentroids[c].data(),
            leafClusterAssignments[c]
        );
        node->leafClusterRepresentatives.push_back(representative);
    }
}


PCNode* PCTree::buildNode(std::vector<int>& vectorIndices,const std::vector<float>& parentPC)
{
    PCNode* node = new PCNode();
    if (static_cast<int>(vectorIndices.size()) <= leafCapacity)
    {
        node->isLeaf = true;
        node->vectorIndices =std::move(vectorIndices);
        node->parentDotProduct =dotProduct(parentPC.data(),parentPC.data());
        populateLeaf(node);
        return node;
    }



    std::vector<float> pc(dim);
    computePCA(vectorIndices, pc);
    node->principalComponent = pc;

    std::vector<std::pair<float, int>> projected;
    projected.reserve(vectorIndices.size());

    for (int idx : vectorIndices)
    {
        float projection =dotProduct(getVector(idx),pc.data());
        projected.emplace_back(projection,idx);
    }

    // Sort vectors according to PCA projection
    std::sort(projected.begin(),projected.end(),
        [](const auto& a, const auto& b)
        {
            return a.first < b.first;
        }
    );



    int actualK =std::min(numPartitions,static_cast<int>(projected.size()));
    std::vector<std::vector<int>> partitions(actualK);

    size_t total = projected.size();

    for (int p = 0; p < actualK; ++p)
    {
        size_t start = (static_cast<size_t>(p) * total)/ actualK;
        size_t end = (static_cast<size_t>(p + 1) * total) / actualK;
        partitions[p].reserve(end - start);
        for (size_t i = start; i < end; ++i)
        {
            partitions[p].push_back(projected[i].second);
        }
    }
    node->splitPoints.clear();

    node->splitPoints.reserve(actualK > 1 ? actualK - 1 : 0);

    for (int p = 1; p < actualK; ++p)
    {
        size_t boundary =(static_cast<size_t>(p) * total)/ actualK;
        node->splitPoints.push_back(projected[boundary].first);
    }

    node->parentDotProduct =dotProduct(parentPC.data(),pc.data());
    node->children.clear();
    node->children.reserve(actualK);

    for (int p = 0; p < actualK; ++p)
    {
        node->children.push_back(buildNode(partitions[p],pc));
    }
    return node;
}


PCNode* PCTree::findLeaf(const float* query) const
{
    if (!root || query == nullptr)
    {
        return nullptr;
    }
    PCNode* current = root;
    while (current != nullptr && !current->isLeaf)
    {
        float queryProj = dotProduct(query, current->principalComponent.data());
        size_t childIndex = 0;
        while (childIndex < current->splitPoints.size() && queryProj >= current->splitPoints[childIndex])
        {
            ++childIndex;
        }

        // Safety check
        if (childIndex >= current->children.size())
        {
            childIndex = current->children.size() - 1;
        }
        current = current->children[childIndex];
    }
    return current;
}


std::vector<int> PCTree::searchNN(const float* query) const
{
    std::vector<int> representatives;
    PCNode* leaf = findLeaf(query);
    if (leaf != nullptr && leaf->representative >= 0)
    {
        representatives.push_back(leaf->representative);
    }
    return representatives;
}


std::vector<int> PCTree::searchNNMulti(const float* query) const
{
    PCNode* leaf = findLeaf(query);
    if (leaf == nullptr)
    {
        return {};
    }
    return leaf->leafClusterRepresentatives;
}


int PCTree::getHeight(PCNode* node) const
{
    if (node == nullptr || node->isLeaf)
        return 0;

    int height = 0;
    for (PCNode* child : node->children)
    {
        height =std::max(height,getHeight(child));
    }
    return 1 + height;
}

int PCTree::getHeight() const
{
    return getHeight(root);
}