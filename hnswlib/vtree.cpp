#include "vtree.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>


VTNode::~VTNode() 
{
    for (const Branch& b : branches) 
    {
        delete b.child; 
    }
}

VoronoiTree::VoronoiTree(char* data_level0_memory,size_t data_size,size_t dim,size_t num_vectors,int numPivots,int leafCapacity)
{
    this->data_level0_memory = data_level0_memory;
    this->data_size = data_size;
    this->dim = dim;
    this->num_vectors = num_vectors;
    this->numPivots = numPivots;
    this->leafCapacity = leafCapacity;

    if (data_level0_memory == nullptr) 
    {
        throw std::invalid_argument("VoronoiTree: data_level0_memory must not be null");
    }
    if (numPivots < 1) 
    {
        throw std::invalid_argument("VoronoiTree: numPivots must be >= 1");
    }
    if (leafCapacity < 1) 
    {
        throw std::invalid_argument("VoronoiTree: leafCapacity must be >= 1");
    }
}

VoronoiTree::~VoronoiTree() 
{
    delete root; 
}
const float* VoronoiTree::getVector(int id) const 
{
    return reinterpret_cast<const float*>(data_level0_memory + static_cast<size_t>(id) * data_size);
}
float VoronoiTree::distance(const float* a, const float* b) const 
{
    float sumSq = 0.0f;
    for (size_t i = 0; i < dim; ++i) 
    {
        const float diff = a[i] - b[i];
        sumSq += diff * diff;
    }
    return sumSq;
}


void VoronoiTree::build() 
{
    delete root;
    root = nullptr;

    if (num_vectors == 0) 
    {
        root = new VTNode();
        root->isLeaf = true;
        return;
    }

    std::vector<int> allIndices(num_vectors);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    root = buildRecursive(std::move(allIndices));
    preprocessLeafEntryPoints(root);
}


void VoronoiTree::choosePivots(const std::vector<int>& indices,std::vector<int>& pivots,std::vector<float>& distanceMatrix) const {
    const int N = static_cast<int>(indices.size());
    const int P = std::min(numPivots, N); 

    pivots.clear();
    pivots.reserve(P);

    distanceMatrix.assign(static_cast<size_t>(P) * static_cast<size_t>(N), 0.0f);
    std::vector<float> nearestDistance(N, std::numeric_limits<float>::max());
    std::vector<bool> isPivot(N, false);

    
    std::vector<const float*> vectorPtrs(N);
    for (int j = 0; j < N; ++j) {
        vectorPtrs[j] = getVector(indices[j]);
    }

 
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> uniform(0, N - 1);

    for (int p = 0; p < P; ++p) 
    {
        int pivotLocal;
        if (p == 0) 
        {
            pivotLocal = uniform(rng);
        } 
        else 
        {
            float best = -1.0f;
            pivotLocal = -1;
            for (int j = 0; j < N; ++j) 
            {
                if (!isPivot[j] && nearestDistance[j] > best) 
                {
                    best = nearestDistance[j];
                    pivotLocal = j;
                }
            }
            if (pivotLocal < 0) {
                pivotLocal = 0;
            }
        }

        isPivot[pivotLocal] = true;
        const int globalIdx = indices[pivotLocal];
        pivots.push_back(globalIdx);

        
        const float* pivotVec = getVector(globalIdx);
        float* row = distanceMatrix.data() + static_cast<size_t>(p) * N;
        for (int j = 0; j < N; ++j) {
            const float d = distance(pivotVec, vectorPtrs[j]);
            row[j] = d;
            if (d < nearestDistance[j]) {
                nearestDistance[j] = d;
            }
        }
    }
}

void VoronoiTree::partition(const std::vector<int>& indices,const std::vector<int>& pivots,const std::vector<float>& distanceMatrix,std::vector<std::vector<int>>& clusters,std::vector<float>& radii) const 
{
    const int P = static_cast<int>(pivots.size());
    const int N = static_cast<int>(indices.size());

    clusters.assign(P,std::vector<int>());
    radii.assign(P, 0.0f);

    
    const size_t estimate = static_cast<size_t>(N) / std::max(P, 1) + 1;
    for (auto& cluster : clusters) 
    {
        cluster.reserve(estimate);
    }

    std::vector<float> bestDist(N);
    std::vector<int> bestPivot(N);

    const float* row0 = distanceMatrix.data();
    for (int j = 0; j < N; ++j) 
    {
        bestDist[j] = row0[j];
        bestPivot[j] = 0;
    }

    for (int p = 1; p < P; ++p) 
    {
        const float* row = distanceMatrix.data() + static_cast<size_t>(p) * N;
        for (int j = 0; j < N; ++j) 
        {
            if (row[j] < bestDist[j]) 
            {
                bestDist[j] = row[j];
                bestPivot[j] = p;
            }
        }
    }

    for (int j = 0; j < N; ++j) 
    {
        const int p = bestPivot[j];
        clusters[p].push_back(indices[j]);
        if (bestDist[j] > radii[p]) 
        {
            radii[p] = bestDist[j];
        }
    }
}


VTNode* VoronoiTree::buildRecursive(std::vector<int>&& indices) 
{
    VTNode* node = new VTNode();

    const int n = static_cast<int>(indices.size());

    if (n <= leafCapacity || numPivots < 2 || n < 2) 
    {
        node->isLeaf = true;
        node->vectorIndices = std::move(indices);
        return node;
    }

    std::vector<int> pivots;
    std::vector<float> distanceMatrix;
    choosePivots(indices, pivots, distanceMatrix);

    std::vector<std::vector<int>> clusters;
    std::vector<float> radii;
    partition(indices, pivots, distanceMatrix, clusters, radii);

    int nonEmpty = 0;
    for (const auto& c : clusters) 
    {
        if (!c.empty()) 
        {
            ++nonEmpty;
        }
    }
    if (nonEmpty <= 1) 
    {
        node->isLeaf = true;
        node->vectorIndices = std::move(indices);
        return node;
    }
    node->isLeaf = false;
    node->branches.reserve(static_cast<size_t>(nonEmpty));

    for (size_t i = 0; i < clusters.size(); ++i) 
    {
        if (clusters[i].empty()) 
        {
            continue; // skip empty cells completely
        }

        Branch branch;
        branch.pivot = pivots[i];
        branch.radius = radii[i];
        branch.child = buildRecursive(std::move(clusters[i]));
        branch.child->parent = node;

        node->branches.push_back(branch);
    }
    return node;
}




std::vector<float> VoronoiTree::computeLeafCentroid(const VTNode* leaf) const
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


int VoronoiTree::findNearestToCentroid(const VTNode* leaf,const std::vector<float>& centroid) const
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
void VoronoiTree::preprocessLeafEntryPoints(VTNode* node)
{
    if (node == nullptr)
    {
        return;
    }
    if (node->isLeaf)
    {
        if (!node->vectorIndices.empty())
        {
            std::vector<float> centroid =computeLeafCentroid(node);

            node->centroidEntryPoint =findNearestToCentroid(node, centroid);
        }
        return;
    }
    for (Branch& branch : node->branches)
    {
        preprocessLeafEntryPoints(branch.child);
    }
}

int VoronoiTree::searchEntryPoint(const float* query) const
{
    if (root == nullptr || query == nullptr)
    {
        return -1;
    }
    const VTNode* current = root;
    while (current != nullptr)
    {
        if (current->isLeaf)
        {
            return current->centroidEntryPoint;
        }
        if (current->branches.empty())
        {
            return -1;
        }
        float bestDist = std::numeric_limits<float>::max();
        const VTNode* bestChild = nullptr;
        for (const Branch& branch : current->branches)
        {
            if (branch.child == nullptr || branch.pivot < 0)
            {
                continue;
            }
            const float d =distance(query, getVector(branch.pivot));
            if (d < bestDist)
            {
                bestDist = d;
                bestChild = branch.child;
            }
        }
        if (bestChild == nullptr)
        {
            return -1;
        }
        current = bestChild;
    }
    return -1;
}

std::vector<int> VoronoiTree::searchNN(const float* query) const
{
    std::vector<int> collected;
    if (root == nullptr || query == nullptr)
    {
        return collected;
    }
    const VTNode* current = root;
    while (current != nullptr)
    {
        // Reached the leaf: return all vectors in the leaf.
        if (current->isLeaf)
        {
            collected.insert(
                collected.end(),
                current->vectorIndices.begin(),
                current->vectorIndices.end()
            );
            break;
        }
        if (current->branches.empty())
        {
            break;
        }
        // Find the pivot closest to the query.
        float bestDist = std::numeric_limits<float>::max();
        const Branch* bestBranch = nullptr;
        for (const Branch& branch : current->branches)
        {
            if (branch.child == nullptr || branch.pivot < 0)
            {
                continue;
            }
            const float d =distance(query, getVector(branch.pivot));
            if (d < bestDist)
            {
                bestDist = d;
                bestBranch = &branch;
            }
        }
        if (bestBranch == nullptr)
        {
            break;
        }
        // Continue through the child of the closest pivot.
        current = bestBranch->child;
    }
    return collected;
}

int VoronoiTree::getHeight() const
{
    return calculateHeight(root);
}

int VoronoiTree::calculateHeight(VTNode* node) const
{
    if (node == nullptr || node->isLeaf)
    {
        return 0;
    }

    int maxChildHeight = 0;
    for (const Branch& b : node->branches)
    {
        maxChildHeight = std::max(maxChildHeight, calculateHeight(b.child));
    }
    return 1 + maxChildHeight;
}