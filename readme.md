
# HNSW Indexing with Tree-Based Candidate Pruning

## What this project does

This project starts with the usual HNSW index and explores whether tree structures can help it search for nearest neighbours more efficiently.

Normally, HNSW uses its graph to move towards the query's nearest neighbours. In this project, we build an additional tree over the same dataset and use that tree to guide the search. We then run the HNSW search variants and compare their results with the standard HNSW implementation.

The trees used in the project are MTree, PCTree, VPTree, and KMeansTree. We also test two optimizations called FINGER and TRI, along with multi-candidate variants for several tree-based methods.

The main files to understand are:

- `combined_test.cpp`: Controls the benchmark. It loads the data, calls the build functions, runs queries, measures performance, and writes the results.
- `hnswlib/hnswalg.h`: Contains the HNSW implementation and the functions that build the trees and execute the different search methods.
- Tree-specific `.cpp` and `.h` files: Contain the implementation details for each tree.

## How the code fits together

The easiest way to understand the project is to follow the calls from `combined_test.cpp` into `hnswalg.h`.

`combined_test.cpp` is the driver. It decides what to run and when, creates the HNSW object, and calls methods on that object.

`hnswalg.h` is where those methods are implemented. It manages the HNSW graph, builds the selected tree structures, and provides the different search functions.

The tree implementation files handle the work specific to each data structure. For example, the HNSW wrapper can call `buildMTree(...)`, while the actual MTree construction logic lives in `mtree.cpp` and `mtree.h`.

In short, the benchmark file coordinates the execution, the HNSW class connects the benchmark to the index and tree implementations, and the tree-specific files implement the individual data structures.

## What happens when the program runs?

### 1. Loading the dataset

When you run the executable, `main()` in `combined_test.cpp` first chooses the dataset directory. If no directory is supplied, it uses `./datasets/gist`.

It then reads three files:

- `base.fvecs`: Contains the vectors that will be indexed.
- `query.fvecs`: Contains the query vectors.
- `groundtruth.ivecs`: Contains the expected nearest neighbours for each query.

The `read_fvecs(...)` and `read_ivecs(...)` helper functions read these binary files into memory. The program checks that the base and query dimensions match before building the index.

### 2. Building the HNSW graph

Next, `main()` creates an L2 distance space and an HNSW object:

```cpp
hnswlib::L2Space space(dim);
hnswlib::HierarchicalNSW<float> hnsw(&space, N, M, ef_construction);
```

It then inserts the base vectors one by one by calling:

```cpp
hnsw.addPoint(...);
```

This call enters the HNSW implementation. Once all the vectors have been inserted, the graph is ready to be used as the baseline and by the tree-guided search methods.

### 3. Preparing FINGER and TRI

Before running the queries, the driver prepares the two search optimizations.

For FINGER, `combined_test.cpp` calls:

```cpp
hnsw.setFingerRank(finger_rank);
hnsw.setUseFinger(true, finger_rank);
hnsw.buildFingerProjections();
```

These calls configure FINGER and build its projection data and supporting information for the indexed nodes.

For TRI, the driver calls:

```cpp
hnsw.buildTriDistances();
```

This precomputes distances for edges in the HNSW graph. The TRI-enabled search functions can then use those stored values during graph traversal.

These preprocessing steps happen before the query loops.

### 4. Building the tree structures

The tree-building calls originate in `combined_test.cpp`, while the corresponding wrapper functions are implemented in `hnswlib/hnswalg.h`.

For example, the driver calls:

```cpp
hnsw.buildPCTree(numPartitions, leaf_capacity, leaf_clusters);
hnsw.buildMTree(mtree_pivots, leaf_capacity, leaf_clusters);
hnsw.buildVantagePointTree(vpt_partitions, leaf_capacity, leaf_clusters);
hnsw.buildKMeansTree(kmeans_clusters, leaf_capacity, leaf_clusters);
```

The general call flow is:

1. `combined_test.cpp` calls the appropriate `hnsw.build...()` method.
2. The method in `hnswalg.h` creates the corresponding tree object using the indexed dataset.
3. The method invokes the tree's build routine where applicable.
4. The tree implementation organises the data into its structure.
5. Control returns to `combined_test.cpp`, which records the build time and, where available, the tree height and memory usage.

For example, `buildMTree(...)` creates an `MTree` object and calls `mtree_->build()`. Similarly, `buildVantagePointTree(...)` creates a `VantagePointTree` and calls `vpt_->build()`. `buildKMeansTree(...)` creates a `KMeansTree` and calls `kmeanstree_->build()`.

PCTree is constructed through `buildPCTree(...)`, with its setup handled by the PCTree implementation.

The driver repeats this process for different values of `numPartitions` and `leaf_capacity`, allowing the benchmark to compare different tree configurations.

### 5. Executing queries

Once the HNSW graph, optimization data, and trees are ready, `combined_test.cpp` iterates over the configured search parameters. For each configuration, it runs the query set through the search methods being evaluated.

The baseline search call looks like this:

```cpp
auto results = hnsw.searchKnn(query, k, nullptr, &profiler);
```

The tree-guided versions use calls such as:

```cpp
hnsw.searchKnnMTree(query, k, nullptr, &profiler, beam_width);
hnsw.searchKnnPCTree(query, k, nullptr, &profiler, beam_width);
hnsw.searchKnnVPTree(query, k, nullptr, &profiler, beam_width);
hnsw.searchKnnKMeansTree(query, k, nullptr, &profiler, beam_width);
```

These search functions are implemented in `hnswalg.h`. The driver does not implement the graph traversal itself. It calls the relevant function and collects the returned neighbours.

The same general pattern applies to FINGER, TRI, and the multi-candidate variants. Depending on the method, the driver calls functions such as `searchKnnFinger(...)`, `searchKnnTri(...)`, or the corresponding tree-specific search functions.

At a high level, a tree-guided query follows this flow:

1. `combined_test.cpp` selects a query vector and calls a search method.
2. The corresponding function in `hnswalg.h` uses the selected tree to guide candidate or starting-point selection.
3. The function continues with its HNSW search logic using the selected candidate or candidates.
4. The function returns the nearest-neighbour results to `combined_test.cpp`.
5. The driver measures elapsed time, collects CPU-cycle information, and compares the returned IDs with ground truth.

The precise candidate-selection and graph-search steps depend on the implementation of each `searchKnn...()` function in `hnswalg.h`.

### 6. Calculating recall

For every query, the driver compares the returned IDs with the expected IDs in `groundtruth.ivecs`.

The helper function `recall_at_k(...)` counts how many returned neighbours appear in the ground-truth top-k set and divides the hit count by the effective value of `k`.

The driver accumulates the per-query recall for each method and calculates the average recall over the processed queries.

### 7. Recording performance metrics

The benchmark records several metrics to compare the different methods.

- **Latency:** The driver measures the time around each search call using `std::chrono`.
- **CPU cycles:** It reads cycle counters around search calls and accumulates the measurements in the profiling object.
- **Distance computations:** The HNSW implementation and profiling code track distance-evaluation workload for methods where counters are enabled.
- **Build time:** The driver measures HNSW construction, FINGER and TRI preprocessing, and tree construction.
- **Memory:** It uses the memory-accounting methods exposed by `hnswalg.h` and the tree classes.

The reported memory values are structural estimates calculated by the program. They are not operating-system resident memory measurements.

## The main function-call map

The following diagram summarises how the files and functions are connected:

```text
combined_test.cpp: main()
    |
    |-- read_fvecs(...) / read_ivecs(...)
    |
    |-- Create L2Space and HierarchicalNSW
    |
    |-- hnsw.addPoint(...)                  [Build HNSW graph]
    |
    |-- hnsw.buildFingerProjections()       [FINGER preprocessing]
    |
    |-- hnsw.buildTriDistances()            [TRI preprocessing]
    |
    |-- hnsw.buildMTree(...)
    |       `-- MTree::build()              [mtree.cpp / mtree.h]
    |
    |-- hnsw.buildPCTree(...)
    |       `-- PCTree implementation       [pctree.cpp / pctree.h]
    |
    |-- hnsw.buildVantagePointTree(...)
    |       `-- VantagePointTree::build()    [vpttree.cpp / vpttree.h]
    |
    |-- hnsw.buildKMeansTree(...)
    |       `-- KMeansTree::build()          [kmeanstree.cpp / kmeanstree.h]
    |
    `-- For each search configuration and query:
            |
            |-- hnsw.searchKnn(...)
            |-- hnsw.searchKnnFinger(...)
            |-- hnsw.searchKnnTri(...)
            |-- hnsw.searchKnnMTree(...) / related variants
            |-- hnsw.searchKnnPCTree(...) / related variants
            |-- hnsw.searchKnnVPTree(...) / related variants
            `-- hnsw.searchKnnKMeansTree(...) / related variants
                    |
                    `-- Record recall, latency, cycles, and other metrics
                        and write benchmark CSV files
```

This diagram shows the main control flow. The actual sequence and set of calls are determined by the loops and enabled variants in `combined_test.cpp`.

## Files worth opening first

If you want to understand or modify the implementation, read the files in this order:

1. **`combined_test.cpp`**: Start with `main()` to see the execution sequence and the search calls.
2. **`hnswlib/hnswalg.h`**: Find the matching `build...()` and `searchKnn...()` functions called by the driver.
3. **The relevant tree implementation:** Open `mtree.cpp`, `pctree.cpp`, `vpttree.cpp`, or `kmeanstree.cpp` to see how the tree is constructed and queried.
4. **`profiling.h`**: Inspect how CPU-cycle and other profiling values are collected.
5. **The generated CSV files:** Use them to compare the methods after running the benchmark.

## Building the project

From the repository root, use the command documented in `compile.txt`. The current example build command is:

```bash
g++ -O3 -std=c++17 -I. -Ihnswlib -I/usr/include/eigen3 \
    combined_test.cpp \
    hnswlib/pctree.cpp \
    hnswlib/vtree.cpp \
    hnswlib/mtree.cpp \
    hnswlib/vpttree.cpp \
    hnswlib/kmeanstree.cpp \
    -o combined_test
```

This command assumes Eigen headers are available at `/usr/include/eigen3`. Adjust the include path if Eigen is installed elsewhere.

## Running the benchmark

Use the default GIST dataset:

```bash
./combined_test
```

Or pass a dataset directory:

```bash
./combined_test ./datasets/gist
./combined_test ./datasets/sift
./combined_test ./datasets/glove
```

Each dataset directory is expected to contain `base.fvecs`, `query.fvecs`, and `groundtruth.ivecs`.

## Output files

The driver writes these CSV files in the current working directory:

- `hnsw_results_gist.csv`: Recall, latency, speedup, and related result summaries.
- `hnsw_distance_computations_gist.csv`: Distance-computation statistics.
- `hnsw_cpu_cycles_gist.csv`: CPU-cycle measurements.
- `hnsw_memory_gist.csv`: Structural memory estimates.


