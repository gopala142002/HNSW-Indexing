#include "pctree.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>


PCTree::PCTree(const char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int leafCapacity,int numPartitions)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->leafCapacity = leafCapacity;
    this->numPartitions = numPartitions;

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


PCNode* PCTree::buildNode(std::vector<int>& vectorIndices,const std::vector<float>& parentPC)
{
    PCNode* node = new PCNode();
    if (static_cast<int>(vectorIndices.size()) <= leafCapacity)
    {
        node->isLeaf = true;
        node->vectorIndices =std::move(vectorIndices);
        node->parentDotProduct =dotProduct(parentPC.data(),parentPC.data());
        std::vector<float> centroid =computeCentroid(node->vectorIndices);

        node->representative =findNearestToPoint(centroid.data(),node->vectorIndices);
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


std::vector<int> PCTree::searchNN(
    const float* query) const
{
    std::vector<int> representatives;
    if (!root || query == nullptr)
    {
        return representatives;
    }
    PCNode* current = root;
    while (current != nullptr)
    {
        if (current->isLeaf)
        {
            if (current->representative >= 0)
            {
                representatives.push_back(current->representative);
            }
            break;
        }
        float queryProj =dotProduct(query,current->principalComponent.data());
        size_t childIndex = 0;
        while (childIndex < current->splitPoints.size() &&queryProj >=current->splitPoints[childIndex])
        {
            ++childIndex;
        }

        // Safety check
        if (childIndex >=current->children.size())
        {
            childIndex =current->children.size() - 1;
        }
        current =current->children[childIndex];
    }
    return representatives;
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