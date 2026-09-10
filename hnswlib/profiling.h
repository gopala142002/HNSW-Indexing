#pragma once

#include <vector>

class profilingStats
{
public:
    std::vector<double> hnswLevelTimeNs;

    double hnswTriLevel0TimeNs = 0.0;
    double hnswFingerLevel0TimeNs = 0.0;

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
    }
};