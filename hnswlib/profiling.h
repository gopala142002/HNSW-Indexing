#pragma once

#include <vector>

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
    }
};