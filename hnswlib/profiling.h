#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

class profilingStats
{
public:
    std::vector<double> hnswLevelTimeNs;

    double hnswTriLevel0TimeNs = 0.0;
    double hnswFingerLevel0TimeNs = 0.0;

    double hnswLevel0_PCTree=0.0;
    double hnswLevel0_MTree=0.0;
    double hnswLevel0_VPTree=0.0;
    double hnswLevel0_KMeanTree=0.0;

    double hnswTriLevel0_PCTree=0.0;
    double hnswTriLevel0_MTree=0.0;
    double hnswTriLevel0_VPTree=0.0;
    double hnswTriLevel0_KMeanTree=0.0;

    double hnswFingerLevel0_PCTree=0.0;
    double hnswFingerLevel0_MTree=0.0;
    double hnswFingerLevel0_VPTree=0.0;
    double hnswFingerLevel0_KMeanTree=0.0;


    double hnswLevel0PCTreeMulti=0.0;
    double hnswFingerLevel0PCTreeMulti=0.0;
    double hnswTriLevel0PCTreeMulti=0.0;

    double hnswLevel0MTreeMulti=0.0;
    double hnswFingerLevel0MTreeMulti=0.0;
    double hnswTriLevel0MTreeMulti=0.0;

    double hnswLevel0VPTreeMulti=0.0;
    double hnswFingerLevel0VPTreeMulti=0.0;
    double hnswTriLevel0VPTreeMulti=0.0;

    double hnswLevel0KMeansTreeMulti=0.0;
    double hnswFingerLevel0KMeansTreeMulti=0.0;
    double hnswTriLevel0KMeansTreeMulti=0.0;
    

    double pctreeTimeNs = 0.0;
    double mtreeTimeNs = 0.0;
    double vptreeTimeNs = 0.0;
    double kmeansTreeTimeNs = 0.0;


    // Timing for the all-levels TRI approach
    double hnswTriAllLevelsUpperLayerTimeNs = 0.0;
    double hnswTriAllLevelsLevel0TimeNs = 0.0;    


    // for count of distance computations
    uint64_t hnswDistanceComputations=0;
    uint64_t hnswFingerDistanceComputations=0;
    uint64_t hnswTriDistanceComputations=0;

    uint64_t hnswDistanceComputations_PCTree=0;
    uint64_t hnswFingerDistanceComputations_PCTree = 0;
    uint64_t hnswTriDistanceComputations_PCTree = 0;

    uint64_t hnswDistanceComputations_MTree = 0;
    uint64_t hnswFingerDistanceComputations_MTree = 0;
    uint64_t hnswTriDistanceComputations_MTree = 0;

    uint64_t hnswDistanceComputations_VPTree = 0;
    uint64_t hnswFingerDistanceComputations_VPTree = 0;
    uint64_t hnswTriDistanceComputations_VPTree = 0;

    uint64_t hnswDistanceComputations_KMeansTree = 0;
    uint64_t hnswFingerDistanceComputations_KMeansTree = 0;
    uint64_t hnswTriDistanceComputations_KMeansTree = 0;

    uint64_t hnswDistanceComputations_PCTreeMulti = 0;
    uint64_t hnswFingerDistanceComputations_PCTreeMulti = 0;
    uint64_t hnswTriDistanceComputations_PCTreeMulti = 0;

    uint64_t hnswDistanceComputations_MTreeMulti = 0;
    uint64_t hnswFingerDistanceComputations_MTreeMulti = 0;
    uint64_t hnswTriDistanceComputations_MTreeMulti = 0;

    uint64_t hnswDistanceComputations_VPTreeMulti = 0;
    uint64_t hnswFingerDistanceComputations_VPTreeMulti = 0;
    uint64_t hnswTriDistanceComputations_VPTreeMulti = 0;

    uint64_t hnswDistanceComputations_KMeansTreeMulti = 0;
    uint64_t hnswFingerDistanceComputations_KMeansTreeMulti = 0;
    uint64_t hnswTriDistanceComputations_KMeansTreeMulti = 0;

    uint64_t pctreeDistanceComputations = 0;
    uint64_t pctreeDistanceComputationsMulti=0;

    uint64_t mtreeDistanceComputations = 0;
    uint64_t mtreeDistanceComputationsMulti = 0;

    uint64_t vptreeDistanceComputations = 0;
    uint64_t vptreeDistanceComputationsMulti = 0;

    uint64_t kmeansTreeDistanceComputations = 0;
    uint64_t kmeansTreeDistanceComputationsMulti = 0;

    uint64_t hnswUpperLayerDistanceComputations = 0;

    uint64_t hnswUpperLayerDistanceComputationsTRI = 0;
    uint64_t hnswLevel0DistanceComputationsWithTRI = 0;

    void reset(size_t num_levels)
    {
        hnswLevelTimeNs.assign(num_levels, 0.0);

        hnswTriLevel0TimeNs = 0.0;
        hnswFingerLevel0TimeNs = 0.0;

        pctreeTimeNs = 0.0;
        mtreeTimeNs = 0.0;
        vptreeTimeNs = 0.0;
        kmeansTreeTimeNs = 0.0;


        hnswLevel0_PCTree=0.0;
        hnswLevel0_MTree=0.0;
        hnswLevel0_VPTree=0.0;
        hnswLevel0_KMeanTree=0.0;

        hnswTriLevel0_PCTree=0.0;
        hnswTriLevel0_MTree=0.0;
        hnswTriLevel0_VPTree=0.0;
        hnswTriLevel0_KMeanTree=0.0;

        hnswFingerLevel0_PCTree=0.0;
        hnswFingerLevel0_MTree=0.0;
        hnswFingerLevel0_VPTree=0.0;
        hnswFingerLevel0_KMeanTree=0.0;

        hnswLevel0PCTreeMulti = 0.0;
        hnswFingerLevel0PCTreeMulti = 0.0;
        hnswTriLevel0PCTreeMulti = 0.0;

        hnswLevel0MTreeMulti = 0.0;
        hnswFingerLevel0MTreeMulti = 0.0;
        hnswTriLevel0MTreeMulti = 0.0;

        hnswLevel0VPTreeMulti = 0.0;
        hnswFingerLevel0VPTreeMulti = 0.0;
        hnswTriLevel0VPTreeMulti = 0.0;

        hnswLevel0KMeansTreeMulti = 0.0;
        hnswFingerLevel0KMeansTreeMulti = 0.0;
        hnswTriLevel0KMeansTreeMulti = 0.0;

        hnswDistanceComputations=0;
        hnswFingerDistanceComputations=0;
        hnswTriDistanceComputations=0;

        hnswDistanceComputations_PCTree=0;
        hnswFingerDistanceComputations_PCTree = 0;
        hnswTriDistanceComputations_PCTree = 0;

        hnswDistanceComputations_MTree = 0;
        hnswFingerDistanceComputations_MTree = 0;
        hnswTriDistanceComputations_MTree = 0;

        hnswTriAllLevelsUpperLayerTimeNs = 0.0;
        hnswTriAllLevelsLevel0TimeNs = 0.0;

        hnswDistanceComputations_VPTree = 0;
        hnswFingerDistanceComputations_VPTree = 0;
        hnswTriDistanceComputations_VPTree = 0;

        hnswDistanceComputations_KMeansTree = 0;
        hnswFingerDistanceComputations_KMeansTree = 0;
        hnswTriDistanceComputations_KMeansTree = 0;

        hnswDistanceComputations_PCTreeMulti = 0;
        hnswFingerDistanceComputations_PCTreeMulti = 0;
        hnswTriDistanceComputations_PCTreeMulti = 0;

        hnswDistanceComputations_MTreeMulti = 0;
        hnswFingerDistanceComputations_MTreeMulti = 0;
        hnswTriDistanceComputations_MTreeMulti = 0;

        hnswDistanceComputations_VPTreeMulti = 0;
        hnswFingerDistanceComputations_VPTreeMulti = 0;
        hnswTriDistanceComputations_VPTreeMulti = 0;

        hnswDistanceComputations_KMeansTreeMulti = 0;
        hnswFingerDistanceComputations_KMeansTreeMulti = 0;
        hnswTriDistanceComputations_KMeansTreeMulti = 0;

        pctreeDistanceComputations = 0;
        mtreeDistanceComputations = 0;
        vptreeDistanceComputations = 0;
        kmeansTreeDistanceComputations = 0;
        hnswUpperLayerDistanceComputations = 0;

        hnswUpperLayerDistanceComputationsTRI=0;
        hnswLevel0DistanceComputationsWithTRI=0;
    }
};