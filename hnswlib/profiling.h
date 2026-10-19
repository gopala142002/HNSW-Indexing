#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cerrno>

#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/perf_event.h>


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





    //CPU CYCLE PROFILING 

    // Base search approaches
    uint64_t hnswTotalCycles = 0;
    uint64_t hnswDistanceCycles = 0;
    uint64_t hnswUpperLayerCycles = 0;
    uint64_t hnswLevel0Cycles = 0;

    uint64_t fingerTotalCycles = 0;
    uint64_t fingerDistanceCycles = 0;
    uint64_t fingerUpperLayerCycles = 0;
    uint64_t fingerLevel0Cycles = 0;

    uint64_t triTotalCycles = 0;
    uint64_t triDistanceCycles = 0;
    uint64_t triUpperLayerCycles = 0;
    uint64_t triLevel0Cycles = 0;


    //PCTREE VARIANTS

    uint64_t hnswPCTreeTotalCycles = 0;
    uint64_t hnswPCTreeDistanceCycles = 0;
    uint64_t hnswPCTreeUpperLayerCycles = 0;
    uint64_t hnswPCTreeLevel0Cycles = 0;

    uint64_t fingerPCTreeTotalCycles = 0;
    uint64_t fingerPCTreeDistanceCycles = 0;
    uint64_t fingerPCTreeUpperLayerCycles = 0;
    uint64_t fingerPCTreeLevel0Cycles = 0;

    uint64_t triPCTreeTotalCycles = 0;
    uint64_t triPCTreeDistanceCycles = 0;
    uint64_t triPCTreeUpperLayerCycles = 0;
    uint64_t triPCTreeLevel0Cycles = 0;


    //MTREE VARIANTS

    uint64_t hnswMTreeTotalCycles = 0;
    uint64_t hnswMTreeDistanceCycles = 0;
    uint64_t hnswMTreeUpperLayerCycles = 0;
    uint64_t hnswMTreeLevel0Cycles = 0;

    uint64_t fingerMTreeTotalCycles = 0;
    uint64_t fingerMTreeDistanceCycles = 0;
    uint64_t fingerMTreeUpperLayerCycles = 0;
    uint64_t fingerMTreeLevel0Cycles = 0;

    uint64_t triMTreeTotalCycles = 0;
    uint64_t triMTreeDistanceCycles = 0;
    uint64_t triMTreeUpperLayerCycles = 0;
    uint64_t triMTreeLevel0Cycles = 0;


    // VPTREE VARIANTS 

    uint64_t hnswVPTreeTotalCycles = 0;
    uint64_t hnswVPTreeDistanceCycles = 0;
    uint64_t hnswVPTreeUpperLayerCycles = 0;
    uint64_t hnswVPTreeLevel0Cycles = 0;

    uint64_t fingerVPTreeTotalCycles = 0;
    uint64_t fingerVPTreeDistanceCycles = 0;
    uint64_t fingerVPTreeUpperLayerCycles = 0;
    uint64_t fingerVPTreeLevel0Cycles = 0;

    uint64_t triVPTreeTotalCycles = 0;
    uint64_t triVPTreeDistanceCycles = 0;
    uint64_t triVPTreeUpperLayerCycles = 0;
    uint64_t triVPTreeLevel0Cycles = 0;

    // KMeans VARIANTS 
    uint64_t hnswKMeansTreeTotalCycles = 0;
    uint64_t hnswKMeansTreeDistanceCycles = 0;
    uint64_t hnswKMeansTreeUpperLayerCycles = 0;
    uint64_t hnswKMeansTreeLevel0Cycles = 0;

    uint64_t fingerKMeansTreeTotalCycles = 0;
    uint64_t fingerKMeansTreeDistanceCycles = 0;
    uint64_t fingerKMeansTreeUpperLayerCycles = 0;
    uint64_t fingerKMeansTreeLevel0Cycles = 0;

    uint64_t triKMeansTreeTotalCycles = 0;
    uint64_t triKMeansTreeDistanceCycles = 0;
    uint64_t triKMeansTreeUpperLayerCycles = 0;
    uint64_t triKMeansTreeLevel0Cycles = 0;


    // PCTree Multi
    uint64_t hnswPCTreeMultiTotalCycles = 0;
    uint64_t hnswPCTreeMultiDistanceCycles = 0;
    uint64_t hnswPCTreeMultiUpperLayerCycles = 0;
    uint64_t hnswPCTreeMultiLevel0Cycles = 0;

    uint64_t fingerPCTreeMultiTotalCycles = 0;
    uint64_t fingerPCTreeMultiDistanceCycles = 0;
    uint64_t fingerPCTreeMultiUpperLayerCycles = 0;
    uint64_t fingerPCTreeMultiLevel0Cycles = 0;

    uint64_t triPCTreeMultiTotalCycles = 0;
    uint64_t triPCTreeMultiDistanceCycles = 0;
    uint64_t triPCTreeMultiUpperLayerCycles = 0;
    uint64_t triPCTreeMultiLevel0Cycles = 0;

    // MTree Multi
    uint64_t hnswMTreeMultiTotalCycles = 0;
    uint64_t hnswMTreeMultiDistanceCycles = 0;
    uint64_t hnswMTreeMultiUpperLayerCycles = 0;
    uint64_t hnswMTreeMultiLevel0Cycles = 0;

    uint64_t fingerMTreeMultiTotalCycles = 0;
    uint64_t fingerMTreeMultiDistanceCycles = 0;
    uint64_t fingerMTreeMultiUpperLayerCycles = 0;
    uint64_t fingerMTreeMultiLevel0Cycles = 0;

    uint64_t triMTreeMultiTotalCycles = 0;
    uint64_t triMTreeMultiDistanceCycles = 0;
    uint64_t triMTreeMultiUpperLayerCycles = 0;
    uint64_t triMTreeMultiLevel0Cycles = 0;

    // VPTree Multi
    uint64_t hnswVPTreeMultiTotalCycles = 0;
    uint64_t hnswVPTreeMultiDistanceCycles = 0;
    uint64_t hnswVPTreeMultiUpperLayerCycles = 0;
    uint64_t hnswVPTreeMultiLevel0Cycles = 0;

    uint64_t fingerVPTreeMultiTotalCycles = 0;
    uint64_t fingerVPTreeMultiDistanceCycles = 0;
    uint64_t fingerVPTreeMultiUpperLayerCycles = 0;
    uint64_t fingerVPTreeMultiLevel0Cycles = 0;

    uint64_t triVPTreeMultiTotalCycles = 0;
    uint64_t triVPTreeMultiDistanceCycles = 0;
    uint64_t triVPTreeMultiUpperLayerCycles = 0;
    uint64_t triVPTreeMultiLevel0Cycles = 0;

    // KMeansTree Multi
    uint64_t hnswKMeansTreeMultiTotalCycles = 0;
    uint64_t hnswKMeansTreeMultiDistanceCycles = 0;
    uint64_t hnswKMeansTreeMultiUpperLayerCycles = 0;
    uint64_t hnswKMeansTreeMultiLevel0Cycles = 0;

    uint64_t fingerKMeansTreeMultiTotalCycles = 0;
    uint64_t fingerKMeansTreeMultiDistanceCycles = 0;
    uint64_t fingerKMeansTreeMultiUpperLayerCycles = 0;
    uint64_t fingerKMeansTreeMultiLevel0Cycles = 0;

    uint64_t triKMeansTreeMultiTotalCycles = 0;
    uint64_t triKMeansTreeMultiDistanceCycles = 0;
    uint64_t triKMeansTreeMultiUpperLayerCycles = 0;
    uint64_t triKMeansTreeMultiLevel0Cycles = 0;

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




        //CPU CYCLE PROFILING 

        // Base search approaches
        hnswTotalCycles = 0;
        hnswDistanceCycles = 0;
        hnswUpperLayerCycles = 0;
        hnswLevel0Cycles = 0;

        fingerTotalCycles = 0;
        fingerDistanceCycles = 0;
        fingerUpperLayerCycles = 0;
        fingerLevel0Cycles = 0;

        triTotalCycles = 0;
        triDistanceCycles = 0;
        triUpperLayerCycles = 0;
        triLevel0Cycles = 0;


        //PCTREE VARIANTS

        hnswPCTreeTotalCycles = 0;
        hnswPCTreeDistanceCycles = 0;
        hnswPCTreeUpperLayerCycles = 0;
        hnswPCTreeLevel0Cycles = 0;

        fingerPCTreeTotalCycles = 0;
        fingerPCTreeDistanceCycles = 0;
        fingerPCTreeUpperLayerCycles = 0;
        fingerPCTreeLevel0Cycles = 0;

        triPCTreeTotalCycles = 0;
        triPCTreeDistanceCycles = 0;
        triPCTreeUpperLayerCycles = 0;
        triPCTreeLevel0Cycles = 0;


        //MTREE VARIANTS

        hnswMTreeTotalCycles = 0;
        hnswMTreeDistanceCycles = 0;
        hnswMTreeUpperLayerCycles = 0;
        hnswMTreeLevel0Cycles = 0;

        fingerMTreeTotalCycles = 0;
        fingerMTreeDistanceCycles = 0;
        fingerMTreeUpperLayerCycles = 0;
        fingerMTreeLevel0Cycles = 0;

        triMTreeTotalCycles = 0;
        triMTreeDistanceCycles = 0;
        triMTreeUpperLayerCycles = 0;
        triMTreeLevel0Cycles = 0;


        // VPTREE VARIANTS 

        hnswVPTreeTotalCycles = 0;
        hnswVPTreeDistanceCycles = 0;
        hnswVPTreeUpperLayerCycles = 0;
        hnswVPTreeLevel0Cycles = 0;

        fingerVPTreeTotalCycles = 0;
        fingerVPTreeDistanceCycles = 0;
        fingerVPTreeUpperLayerCycles = 0;
        fingerVPTreeLevel0Cycles = 0;

        triVPTreeTotalCycles = 0;
        triVPTreeDistanceCycles = 0;
        triVPTreeUpperLayerCycles = 0;
        triVPTreeLevel0Cycles = 0;

        // KMeans VARIANTS 
        hnswKMeansTreeTotalCycles = 0;
        hnswKMeansTreeDistanceCycles = 0;
        hnswKMeansTreeUpperLayerCycles = 0;
        hnswKMeansTreeLevel0Cycles = 0;

        fingerKMeansTreeTotalCycles = 0;
        fingerKMeansTreeDistanceCycles = 0;
        fingerKMeansTreeUpperLayerCycles = 0;
        fingerKMeansTreeLevel0Cycles = 0;

        triKMeansTreeTotalCycles = 0;
        triKMeansTreeDistanceCycles = 0;
        triKMeansTreeUpperLayerCycles = 0;
        triKMeansTreeLevel0Cycles = 0;


        // PCTree Multi
        hnswPCTreeMultiTotalCycles = 0;
        hnswPCTreeMultiDistanceCycles = 0;
        hnswPCTreeMultiUpperLayerCycles = 0;
        hnswPCTreeMultiLevel0Cycles = 0;

        fingerPCTreeMultiTotalCycles = 0;
        fingerPCTreeMultiDistanceCycles = 0;
        fingerPCTreeMultiUpperLayerCycles = 0;
        fingerPCTreeMultiLevel0Cycles = 0;

        triPCTreeMultiTotalCycles = 0;
        triPCTreeMultiDistanceCycles = 0;
        triPCTreeMultiUpperLayerCycles = 0;
        triPCTreeMultiLevel0Cycles = 0;

        // MTree Multi
        hnswMTreeMultiTotalCycles = 0;
        hnswMTreeMultiDistanceCycles = 0;
        hnswMTreeMultiUpperLayerCycles = 0;
        hnswMTreeMultiLevel0Cycles = 0;

        fingerMTreeMultiTotalCycles = 0;
        fingerMTreeMultiDistanceCycles = 0;
        fingerMTreeMultiUpperLayerCycles = 0;
        fingerMTreeMultiLevel0Cycles = 0;

        triMTreeMultiTotalCycles = 0;
        triMTreeMultiDistanceCycles = 0;
        triMTreeMultiUpperLayerCycles = 0;
        triMTreeMultiLevel0Cycles = 0;

        // VPTree Multi
        hnswVPTreeMultiTotalCycles = 0;
        hnswVPTreeMultiDistanceCycles = 0;
        hnswVPTreeMultiUpperLayerCycles = 0;
        hnswVPTreeMultiLevel0Cycles = 0;

        fingerVPTreeMultiTotalCycles = 0;
        fingerVPTreeMultiDistanceCycles = 0;
        fingerVPTreeMultiUpperLayerCycles = 0;
        fingerVPTreeMultiLevel0Cycles = 0;

        triVPTreeMultiTotalCycles = 0;
        triVPTreeMultiDistanceCycles = 0;
        triVPTreeMultiUpperLayerCycles = 0;
        triVPTreeMultiLevel0Cycles = 0;

        // KMeansTree Multi
        hnswKMeansTreeMultiTotalCycles = 0;
        hnswKMeansTreeMultiDistanceCycles = 0;
        hnswKMeansTreeMultiUpperLayerCycles = 0;
        hnswKMeansTreeMultiLevel0Cycles = 0;

        fingerKMeansTreeMultiTotalCycles = 0;
        fingerKMeansTreeMultiDistanceCycles = 0;
        fingerKMeansTreeMultiUpperLayerCycles = 0;
        fingerKMeansTreeMultiLevel0Cycles = 0;

        triKMeansTreeMultiTotalCycles = 0;
        triKMeansTreeMultiDistanceCycles = 0;
        triKMeansTreeMultiUpperLayerCycles = 0;
        triKMeansTreeMultiLevel0Cycles = 0;
    }

    public:
    void startCpuCycleCounter()
    {
        if (cpuCyclesFd_ != -1)
            throw std::runtime_error("CPU cycle counter already started.");

        #if !defined(__i386__) && !defined(__x86_64__)
            throw std::runtime_error(
                "RDPMC CPU-cycle reader requires an x86 processor.");
        #else
            perf_event_attr attr{};
            attr.type = PERF_TYPE_HARDWARE;
            attr.size = sizeof(attr);
            attr.config = PERF_COUNT_HW_CPU_CYCLES;
            attr.disabled = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv = 1;

            cpuCyclesFd_ = static_cast<int>(syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0));

            if (cpuCyclesFd_ == -1)
            {
                throw std::runtime_error(std::string("perf_event_open failed: ") + std::strerror(errno));
            }

            long pageSize = sysconf(_SC_PAGESIZE);
            if (pageSize <= 0)
            {
                cleanupCpuCycleCounter();
                throw std::runtime_error("Unable to get system page size.");
            }   

            cpuCyclesPageSize_ = static_cast<size_t>(pageSize);
            void* mapped = mmap(nullptr,cpuCyclesPageSize_,PROT_READ,MAP_SHARED,cpuCyclesFd_,0);

            if (mapped == MAP_FAILED)
            {
                cpuCyclesPage_ = nullptr;
                cleanupCpuCycleCounter();
                throw std::runtime_error("Unable to mmap CPU cycle counter.");
            }

            cpuCyclesPage_ =static_cast<perf_event_mmap_page*>(mapped);

            if (!cpuCyclesPage_->cap_user_rdpmc)
            {
                cleanupCpuCycleCounter();
                throw std::runtime_error(
                    "Kernel does not permit user-space RDPMC reads.");
            }

            if (ioctl(cpuCyclesFd_, PERF_EVENT_IOC_RESET, 0) == -1 || ioctl(cpuCyclesFd_, PERF_EVENT_IOC_ENABLE, 0) == -1)
            {
                cleanupCpuCycleCounter();
                throw std::runtime_error("Unable to reset or enable CPU cycle counter.");
            }

            cpuCyclesRunning_ = true;
        #endif
    }

    uint64_t readCpuCycles() const
    {
        if (cpuCyclesFd_ == -1 || cpuCyclesPage_ == nullptr)
            throw std::runtime_error(
                "CPU cycle counter has not been started.");

        #if defined(__i386__) || defined(__x86_64__)
            for (;;)
            {
                const uint32_t seq = __atomic_load_n(
                    &cpuCyclesPage_->lock, __ATOMIC_ACQUIRE);

                if (seq & 1)
                    continue;

                const uint32_t index = cpuCyclesPage_->index;
                int64_t count = cpuCyclesPage_->offset;
                const uint16_t width = cpuCyclesPage_->pmc_width;

                if (index != 0)
                {
                    uint32_t low, high;

                    asm volatile("rdpmc": "=a"(low), "=d"(high): "c"(index - 1));

                    uint64_t pmc = (static_cast<uint64_t>(high) << 32) | low;

                    if (width == 0)
                        throw std::runtime_error("Invalid hardware counter width.");

                    if (width < 64)
                    {
                        const int shift = 64 - width;
                        pmc = static_cast<uint64_t>(static_cast<int64_t>(pmc << shift) >> shift);
                    }
                    count += static_cast<int64_t>(pmc);
                }

                __atomic_thread_fence(__ATOMIC_ACQUIRE);

                if (seq == __atomic_load_n(&cpuCyclesPage_->lock, __ATOMIC_RELAXED))
                {
                    return static_cast<uint64_t>(count);
                }
            }
        #endif
    }

    void stopCpuCycleCounter()
    {
        if (cpuCyclesFd_ != -1 && cpuCyclesRunning_)
        {
            if (ioctl(cpuCyclesFd_, PERF_EVENT_IOC_DISABLE, 0) == -1)
                throw std::runtime_error(
                    "Unable to disable CPU cycle counter.");

            cpuCyclesRunning_ = false;
        }
    }

    ~profilingStats()
    {
        cleanupCpuCycleCounter();
    }

    private:
    int cpuCyclesFd_ = -1;
    perf_event_mmap_page* cpuCyclesPage_ = nullptr;
    size_t cpuCyclesPageSize_ = 0;
    bool cpuCyclesRunning_ = false;

    void cleanupCpuCycleCounter() noexcept
    {
        if (cpuCyclesFd_ != -1)
        {
            if (cpuCyclesRunning_)
                (void)ioctl(
                    cpuCyclesFd_, PERF_EVENT_IOC_DISABLE, 0);

            cpuCyclesRunning_ = false;

            if (cpuCyclesPage_ != nullptr)
            {
                munmap(cpuCyclesPage_, cpuCyclesPageSize_);
                cpuCyclesPage_ = nullptr;
            }

            close(cpuCyclesFd_);
            cpuCyclesFd_ = -1;
        }
    }
};