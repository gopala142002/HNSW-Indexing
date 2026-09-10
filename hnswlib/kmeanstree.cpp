#include "kmeanstree.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>


KMeansTree::KMeansTree(const char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int numClusters,int leafCapacity,int leafClusters)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->numClusters = numClusters;
    this->leafCapacity = leafCapacity;
    this->leafClusters = leafClusters;
    if (data_level0_memory == nullptr)
    {
        throw std::invalid_argument("KMeansTree: data_level0_memory must not be null");
    }
    if (dim == 0)
    {
        throw std::invalid_argument("KMeansTree: dimension must be greater than 0");
    }
    if (numClusters < 2)
    {
        throw std::invalid_argument("KMeansTree: numClusters must be >= 2");
    }
    if (leafClusters < 1)
    {
        throw std::invalid_argument("KMeansTree: leafClusters must be >= 1");
    }
    if (leafCapacity < 1)
    {
        throw std::invalid_argument("KMeansTree: leafCapacity must be >= 1");
    }
}


KMeansTree::~KMeansTree()
{
    delete root;
}


void KMeansTree::build()
{
    delete root;
    root = nullptr;
    if (num_vectors == 0)
    {
        root = new KMeansNode();
        root->isLeaf = true;
        return;
    }

    std::vector<int> allIndices(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i)
    {
        allIndices[i] = static_cast<int>(i);
    }

    root = buildRecursive(std::move(allIndices));
}


float KMeansTree::squaredDistance(const float* a,const float* b) const
{
    float result = 0.0f;
    for (size_t d = 0; d < dim; ++d)
    {
        float diff = a[d] - b[d];
        result += diff * diff;
    }
    return result;
}


float KMeansTree::squaredDistance(const std::vector<float>& a,const float* b) const
{
    float result = 0.0f;
    for (size_t d = 0; d < dim; ++d)
    {
        float diff = a[d] - b[d];
        result += diff * diff;
    }
    return result;
}


int KMeansTree::findNearestCentroid(const float* vector,const std::vector<std::vector<float>>& centroids) const
{
    int nearest = 0;
    float minDistance =squaredDistance(centroids[0],vector);
    for (size_t c = 1; c < centroids.size(); ++c)
    {
        float distance =squaredDistance(centroids[c],vector);
        if (distance < minDistance)
        {
            minDistance = distance;
            nearest = static_cast<int>(c);
        }
    }
    return nearest;
}


void KMeansTree::initializeCentroids(const std::vector<int>& vectorIndices,int k,std::vector<std::vector<float>>& centroids) const
{
    const int n =static_cast<int>(vectorIndices.size());
    k =std::min(k, n);
    centroids.clear();
    centroids.resize(k,std::vector<float>(dim, 0.0f));

    for (int c = 0; c < k; ++c)
    {
        int position =(static_cast<long long>(c) * n) / k;
        if (position >= n)
        {
            position = n - 1;
        }
        const float* vec =getVector(vectorIndices[position]);
        for (size_t d = 0; d < dim; ++d)
        {
            centroids[c][d] = vec[d];
        }
    }
}


void KMeansTree::runKMeans(const std::vector<int>& vectorIndices,std::vector<std::vector<float>>& centroids,std::vector<std::vector<int>>& clusters) const
{
    const int k =static_cast<int>(centroids.size());
    const int n =static_cast<int>(vectorIndices.size());
    if (k == 0 || n == 0)
    {
        clusters.clear();
        return;
    }

    std::vector<int> assignments(n, -1);

    for (int iteration = 0;iteration < MAX_KMEANS_ITERATIONS;++iteration)
    {
        clusters.assign(k,std::vector<int>());
        for (int i = 0; i < n; ++i)
        {
            int vectorID =vectorIndices[i];
            int clusterID =findNearestCentroid(getVector(vectorID),centroids);
            assignments[i] = clusterID;
            clusters[clusterID].push_back(vectorID);
        }

        std::vector<std::vector<float>> newCentroids(k,std::vector<float>(dim, 0.0f));
        std::vector<int> clusterCounts(k, 0);
        for (int i = 0; i < n; ++i)
        {
            int clusterID =assignments[i];

            const float* vec =getVector(vectorIndices[i]);
            clusterCounts[clusterID]++;
            for (size_t d = 0; d < dim; ++d)
            {
                newCentroids[clusterID][d] +=vec[d];
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
                    int assigned =assignments[i];
                    float distance =squaredDistance(getVector(vectorIndices[i]),centroids[assigned].data());
                    if (distance > farthestDistance)
                    {
                        farthestDistance = distance;
                        farthestIndex = i;
                    }
                }

                if (farthestIndex >= 0)
                {
                    const float* vec =getVector(vectorIndices[farthestIndex]);
                    for (size_t d = 0; d < dim; ++d)
                    {
                        newCentroids[c][d] =vec[d];
                    }
                    clusterCounts[c] = 1;
                }
            }
        }

        for (int c = 0; c < k; ++c)
        {
            if (clusterCounts[c] > 0)
            {
                float invCount =1.0f /static_cast<float>(clusterCounts[c]);
                for (size_t d = 0; d < dim; ++d)
                {
                    newCentroids[c][d] *=invCount;
                }
            }
        }

        float maxMovement = 0.0f;
        for (int c = 0; c < k; ++c)
        {
            float movement = 0.0f;
            for (size_t d = 0; d < dim; ++d)
            {
                float diff =newCentroids[c][d] -centroids[c][d];
                movement += diff * diff;
            }
            maxMovement =std::max(maxMovement,movement);
        }

        centroids =std::move(newCentroids);
        if (maxMovement <KMEANS_TOLERANCE *KMEANS_TOLERANCE)
        {
            break;
        }
    }
    // Final assignment using final centroids
    clusters.assign(k,std::vector<int>());
    for (int vectorID : vectorIndices)
    {
        int clusterID =findNearestCentroid(getVector(vectorID),centroids);
        clusters[clusterID].push_back(vectorID);
    }
}


int KMeansTree::findNearestToPoint(const std::vector<float>& point,const std::vector<int>& vectorIndices) const
{
    if (vectorIndices.empty())
    {
        return -1;
    }
    int nearest =vectorIndices[0];
    float minDistance =squaredDistance(point,getVector(nearest));
    for (int vectorID : vectorIndices)
    {
        float distance =squaredDistance(point,getVector(vectorID));
        if (distance < minDistance)
        {
            minDistance = distance;
            nearest = vectorID;
        }
    }

    return nearest;
}


void KMeansTree::populateLeaf(KMeansNode* node) const
{
    //Approach 1: single leaf centroid / representative
    std::vector<float> centroid(dim, 0.0f);
    if (!node->vectorIndices.empty())
    {
        for (int vectorID : node->vectorIndices)
        {
            const float* vec = getVector(vectorID);
            for (size_t d = 0; d < dim; ++d)
            {
                centroid[d] += vec[d];
            }
        }
        float invCount = 1.0f / static_cast<float>(node->vectorIndices.size());
        for (size_t d = 0; d < dim; ++d)
        {
            centroid[d] *= invCount;
        }
    }
    node->leafCentroid = centroid;
    node->representative = findNearestToPoint(centroid, node->vectorIndices);

    //Approach 2: leaf-level K-Means for multiple HNSW entry points 
    std::vector<std::vector<float>> leafCentroids;
    std::vector<std::vector<int>> leafClusterAssignments;

    initializeCentroids(node->vectorIndices, this->leafClusters, leafCentroids);
    runKMeans(node->vectorIndices, leafCentroids, leafClusterAssignments);

    node->leafClusterCentroids = std::move(leafCentroids);

    node->leafClusterRepresentatives.clear();
    node->leafClusterRepresentatives.reserve(node->leafClusterCentroids.size());
    for (size_t c = 0; c < node->leafClusterCentroids.size(); ++c)
    {
        int representative = findNearestToPoint(
            node->leafClusterCentroids[c],
            leafClusterAssignments[c]
        );
        node->leafClusterRepresentatives.push_back(representative);
    }
}


KMeansNode* KMeansTree::buildRecursive(std::vector<int>&& vectorIndices)
{
    KMeansNode* node =new KMeansNode();
    const int n =static_cast<int>(vectorIndices.size());
    if (n <= leafCapacity)
    {
        node->isLeaf = true;
        node->vectorIndices =std::move(vectorIndices);
        populateLeaf(node);
        return node;
    }

    std::vector<std::vector<float>> centroids;

    initializeCentroids(vectorIndices,numClusters,centroids);
    std::vector<std::vector<int>> clusters;
    runKMeans(vectorIndices,centroids,clusters);

    int nonEmptyClusters = 0;
    for (const auto& cluster : clusters)
    {
        if (!cluster.empty())
        {
            ++nonEmptyClusters;
        }
    }
   
    if (nonEmptyClusters <= 1)
    {
        node->isLeaf = true;
        node->vectorIndices =std::move(vectorIndices);
        populateLeaf(node);
        return node;
    }
    node->isLeaf = false;
    node->centroids.clear();
    node->children.clear();
    node->centroids.reserve(nonEmptyClusters);
    node->children.reserve(nonEmptyClusters);
    for (size_t c = 0;c < clusters.size();++c)
    {
        if (clusters[c].empty())
        {
            continue;
        }
        node->centroids.push_back(centroids[c]);
        node->children.push_back(buildRecursive(std::move(clusters[c])));
    }
    return node;
}


KMeansNode* KMeansTree::findLeaf(const float* query) const
{
    if (root == nullptr || query == nullptr)
    {
        return nullptr;
    }
    KMeansNode* current = root;
    while (current != nullptr && !current->isLeaf)
    {
        int nearestChild = findNearestCentroid(query, current->centroids);
        if (nearestChild < 0 || nearestChild >= static_cast<int>(current->children.size()))
        {
            return nullptr;
        }
        current = current->children[nearestChild];
    }
    return current;
}

int KMeansTree::searchNN(const float* query) const
{
    KMeansNode* leaf = findLeaf(query);
    return leaf ? leaf->representative : -1;
}

std::vector<int> KMeansTree::searchNNMulti(const float* query) const
{
    KMeansNode* leaf = findLeaf(query);
    if (leaf == nullptr)
    {
        return {};
    }
    return leaf->leafClusterRepresentatives;
}

int KMeansTree::calculateHeight(KMeansNode* node) const
{
    if (node == nullptr || node->isLeaf)
    {
        return 0;
    }
    int height = 0;
    for (KMeansNode* child :node->children)
    {
        height =std::max(height,calculateHeight(child));
    }
    return 1 + height;
}

int KMeansTree::getHeight() const
{
    return calculateHeight(root);
}