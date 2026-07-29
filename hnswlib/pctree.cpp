#include "pctree.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <queue>

PCTree::PCTree(const char* data_level0_memory, size_t data_size, size_t dim,size_t num_vectors, int leafCapacity)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->leafCapacity = leafCapacity;

    std::vector<int> allIndices(num_vectors);
    for(int i=0;i<num_vectors;i++)
    {
        allIndices[i]=i;
    }
    std::vector<float> initialPC(dim, 0.0f);
    initialPC[0] = 1.0f;
    
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
    std::vector<float> centroid = computeCentroid(vectorIndices);

    if (pc.empty() || pc[0] == 0.0f) 
    {
        pc.assign(dim, 1.0f / std::sqrt((float)dim));
    }
    const int PC_itr = 20;
    const float acceptable_error = 1e-6f;
    
    std::vector<float> next_pc(dim);
    std::vector<float> centered(dim);
    
    for (int iter = 0; iter < PC_itr; ++iter) 
    {
        std::fill(next_pc.begin(), next_pc.end(), 0.0f);
        
        for (int idx : vectorIndices) 
        {
            const float* vec = getVector(idx);
            
            for (size_t d = 0; d < dim; ++d) 
            {
                centered[d] = vec[d] - centroid[d];
            }
    
            float projection = 0.0f;
            for (size_t d = 0; d < dim; ++d) 
            {
                projection += centered[d] * pc[d];
            }
            
            for (size_t d = 0; d < dim; ++d) 
            {
                next_pc[d] += projection * centered[d];
            }
        }
        
        float norm = 0.0f;
        for (size_t d = 0; d < dim; ++d) 
        {
            norm += next_pc[d] * next_pc[d];
        }
        
        if (norm < 1e-10f) 
        {
            break;
        }
        
        norm = std::sqrt(norm);
    
        float dot_diff = 0.0f;
        for (size_t d = 0; d < dim; ++d) 
        {
            float normalized = next_pc[d] / norm;
            dot_diff += std::abs(normalized - pc[d]);
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
        return -1;

    int nearest = vectorIndices[0];
    float minDist = squaredL2Distance(point, getVector(nearest));
    for (int idx : vectorIndices) 
    {
        float dist = squaredL2Distance(point, getVector(idx));
        if (dist < minDist) 
        {
            minDist = dist;
            nearest = idx;
        }
    }
    return nearest;
}

PCNode* PCTree::buildNode(std::vector<int> &vectorIndices,const std::vector<float>& parentPC) 
{    
    PCNode* node = new PCNode();  

    if ((int)vectorIndices.size() <= leafCapacity) 
    {
        node->isLeaf = true;
        node->vectorIndices = std::move(vectorIndices);
        node->parentDotProduct = dotProduct(parentPC.data(), parentPC.data());
        
        std::vector<float> centroid = computeCentroid(node->vectorIndices);
        node->representative = findNearestToPoint(centroid.data(),node->vectorIndices);
        
        return node;
    }
    
    std::vector<float> pc(dim);
    computePCA(vectorIndices, pc);
    node->principalComponent = pc;
    

    std::vector<float> projections;
    projections.reserve(vectorIndices.size());
    
    for (int idx : vectorIndices) 
    {
        float proj = dotProduct(getVector(idx), pc.data());
        projections.push_back(proj);
    }
    
    std::vector<float> sortedProj = projections;
    std::nth_element(sortedProj.begin(),sortedProj.begin() + sortedProj.size()/2,sortedProj.end());
    node->medianProjection = sortedProj[sortedProj.size() / 2];
    
    std::vector<int> leftIndices, rightIndices;
    leftIndices.reserve(vectorIndices.size());
    rightIndices.reserve(vectorIndices.size());
    
    for (size_t i = 0; i < vectorIndices.size(); ++i) 
    {
        if (projections[i] <= node->medianProjection) 
        {
            leftIndices.push_back(vectorIndices[i]);
        } 
        else 
        {
            rightIndices.push_back(vectorIndices[i]);
        }
    }
    
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

    node->parentDotProduct = dotProduct(parentPC.data(), pc.data());

    node->left = buildNode(leftIndices, pc);
    node->right = buildNode(rightIndices, pc);
    return node;
}

std::vector<int> PCTree::searchNN(const float* query) const 
{
    std::vector<int> representatives;
    if (!root) 
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
        float queryProj = dotProduct(query, current->principalComponent.data());
        current = (queryProj <= current->medianProjection) ? current->left : current->right;
    }
    return representatives;
}

int PCTree::getHeight(PCNode* root) const
{
    if (root == nullptr || root->isLeaf)
    {
        return 0;
    }
    int leftHeight = getHeight(root->left);
    int rightHeight = getHeight(root->right);
    return 1 + std::max(leftHeight, rightHeight);
}
int PCTree::getHeight() const
{
    return getHeight(root);
}
