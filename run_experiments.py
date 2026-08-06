import os
import subprocess
import csv
import json
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).parent.resolve()
BUILD_DIR = WORKSPACE_ROOT / "build"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
RESULTS_DIR = WORKSPACE_ROOT / "results_experiments"

RESULTS_DIR.mkdir(exist_ok=True)
BUILD_DIR.mkdir(exist_ok=True)

CSV_FIELDNAMES = [
    "structure",
    "dataset",
    "param_M",
    "param_ef_construction",
    "param_ef_search",
    "param_leafCapacity",
    "param_numPivots",
    "metric_recall",
    "metric_build_time",
    "metric_latency",
    "metric_height",
    "metric_speedup"
]

PARAM_CONFIGS = {
    "hnsw": {
        "test_file": "test_hnsw.cpp",
        "binary": "test_hnsw",
        "parameters": {
            "M": [4, 8, 16, 32],
            "ef_construction": [50, 100, 200, 400],
            "ef_search": [20, 50, 100, 200]
        },
        "arg_order": ["M", "ef_construction", "ef_search"]
    },
    "mtree": {
        "test_file": "test_mtree.cpp",
        "binary": "test_mtree",
        "parameters": {
            "M": [4, 8, 16, 32],
            "ef_construction": [50, 100, 200, 400],
            "ef_search": [20, 50, 100, 200]
        },
        "arg_order": ["M", "ef_construction", "ef_search"]
    },
    "vpttree": {
        "test_file": "test_vpttree.cpp",
        "binary": "test_vpttree",
        "parameters": {
            "M": [4, 8, 16, 32],
            "ef_construction": [50, 100, 200],
            "ef_search": [20, 50, 100],
            "leafCapacity": [32, 64, 128]
        },
        "arg_order": ["M", "ef_construction", "ef_search", "leafCapacity"]
    },
    "pctree": {
        "test_file": "test_pctree.cpp",
        "binary": "test_pctree",
        "parameters": {
            "M": [4, 8, 16, 32],
            "ef_construction": [50, 100, 200],
            "ef_search": [20, 50, 100],
            "leafCapacity": [32, 64, 128]
        },
        "arg_order": ["M", "ef_construction", "ef_search", "leafCapacity"]
    },
    "vtree": {
        "test_file": "test_vtree.cpp",
        "binary": "test_vtree",
        "parameters": {
            "M": [4, 8, 16, 32],
            "ef_construction": [50, 100, 200],
            "ef_search": [20, 50, 100],
            "numPivots": [4, 8, 16],
            "leafCapacity": [32, 64, 128]
        },
        "arg_order": ["M", "ef_construction", "ef_search", "numPivots", "leafCapacity"]
    }
}

DATASETS = {
    "sift": "sift",
    "glove": "glove",
    "gist": "gist"
}


def log(msg, end="\n"):
    """Thread/Progress safe print logging."""
    try:
        from tqdm import tqdm
        tqdm.write(f"{msg}", end=end)
    except ImportError:
        print(f"{msg}", end=end, flush=True)


def build_project():
    """Compiles C++ sources into binaries using g++."""
    log("Checking and building binaries with g++...")
    
    compile_commands = [
        ("test_hnsw", "g++ -O3 -std=c++17 -I. test_hnsw.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o build/test_hnsw"),
        ("test_mtree", "g++ -O3 -std=c++17 -I. test_mtree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o build/test_mtree"),
        ("test_vpttree", "g++ -O3 -std=c++17 -I. test_vpttree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o build/test_vpttree"),
        ("test_pctree", "g++ -O3 -std=c++17 -I. test_pctree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o build/test_pctree"),
        ("test_vtree", "g++ -O3 -std=c++17 -I. test_vtree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o build/test_vtree"),
    ]
    
    try:
        for binary_name, cmd in compile_commands:
            log(f"  Compiling: {binary_name}...")
            subprocess.run(
                cmd,
                shell=True,
                cwd=WORKSPACE_ROOT,
                check=True,
                capture_output=True,
                text=True
            )
        log("✓ Binaries building successful!")
        return True
    except subprocess.CalledProcessError as e:
        log(f"✗ Build failed for command: {e.cmd}")
        log(f"  Error details:\n{e.stderr}")
        return False


def run_test(binary_path, dataset_path, params_list):
    """Executes a single test binary with supplied dataset and parameters."""
    try:
        cmd = [str(binary_path), str(dataset_path)] + [str(p) for p in params_list]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            log(f"  ✗ Binary exited with error code {result.returncode}: {result.stderr.strip()}")
            return None
        return result.stdout
    except Exception as e:
        log(f"  ✗ ERROR running test: {e}")
        return None


def parse_output(output):
    """Extracts performance metrics from standard stdout using regex patterns."""
    if not output:
        return None
    
    metrics = {}
    
    patterns = {
        'recall': r'(?:Recall|recall)\s*(?:\d+)?\s*[:@=]+\s*([0-9.]+)',
        'build_time': r'(?:built in|Build time)[:\s]+([0-9.]+)\s*s',
        'latency': r'(?:latency|Latency)[:\s]+([0-9.]+)\s*(?:us|µs|ms)?',
        'height': r'(?:height|Height)[:\s]+(\d+)',
        'speedup': r'(?:Speedup|speedup)[:\s/()]+([0-9.]+)',
    }
    
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                metrics[metric_name] = float(matches[-1])
            except ValueError:
                pass
    
    return metrics if metrics else None


def generate_parameter_combinations(params_dict):
    """Generates a Cartesian product of parameter options."""
    import itertools
    
    param_names = list(params_dict.keys())
    param_values = [params_dict[name] for name in param_names]
    
    combinations = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)
    
    return combinations


def run_all_experiments(structures_to_run=None, datasets_to_run=None, dry_run=False):
    """Runs tests across all combinations of structures, configs, and datasets."""
    log("=" * 80)
    log("STARTING COMPREHENSIVE EXPERIMENT SUITE")
    log("=" * 80)
    
    structures_to_run = [s for s in (structures_to_run or PARAM_CONFIGS.keys()) if s in PARAM_CONFIGS]
    datasets_to_run = [d for d in (datasets_to_run or DATASETS.keys()) if d in DATASETS]
    
    all_results = []
    
    if not dry_run and not build_project():
        log("Build failed. Aborting.")
        return all_results
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        def tqdm(iterable, **kwargs): return iterable

    csv_f = open(csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
    csv_writer.writeheader()
    csv_f.flush()
    
    log(f"Writing incremental CSV to: {csv_file}\n")
    
    try:
        struct_iter = tqdm(structures_to_run, desc="Structures", colour="green") if has_tqdm else structures_to_run
        for struct_name in struct_iter:
            struct_config = PARAM_CONFIGS[struct_name]
            log(f"\n--- Testing Structure: {struct_name.upper()} ---")
            
            param_combos = generate_parameter_combinations(struct_config["parameters"])
            log(f"Total parameter combinations: {len(param_combos)}")
            
            combo_iter = tqdm(param_combos, desc=f"  {struct_name} configs", leave=False) if has_tqdm else param_combos
            for combo_idx, param_combo in enumerate(combo_iter, 1):
                arg_order = struct_config["arg_order"]
                param_args = [param_combo[param_name] for param_name in arg_order]
                
                dataset_iter = tqdm(datasets_to_run, desc="    datasets", leave=False) if has_tqdm else datasets_to_run
                for dataset in dataset_iter:
                    dataset_path = DATASETS_DIR / DATASETS[dataset]
                    
                    if not dataset_path.exists():
                        log(f"  ✗ Dataset directory not found: {dataset_path}")
                        continue
                    
                    binary_path = BUILD_DIR / struct_config["binary"]
                    if not binary_path.exists():
                        binary_path = WORKSPACE_ROOT / struct_config["binary"]
                    
                    if not dry_run and not binary_path.exists():
                        log(f"  ✗ Binary file not found: {binary_path}")
                        continue
                    
                    if dry_run:
                        log(f"  [DRY-RUN] Would execute {binary_path.name} on {dataset} with {param_args}")
                        continue
                        
                    output = run_test(binary_path, dataset_path, param_args)
                    
                    if output:
                        metrics = parse_output(output)
                        if metrics:
                            result = {
                                "structure": struct_name,
                                "dataset": dataset,
                                "parameters": param_combo,
                                "metrics": metrics
                            }
                            all_results.append(result)
                            
                            # Formulate CSV Row
                            row = {field: "" for field in CSV_FIELDNAMES}
                            row["structure"] = struct_name
                            row["dataset"] = dataset
                            for param_name, param_value in param_combo.items():
                                row[f"param_{param_name}"] = param_value
                            for metric_name, metric_value in metrics.items():
                                row[f"metric_{metric_name}"] = metric_value
                            
                            csv_writer.writerow(row)
                            csv_f.flush()
                            os.fsync(csv_f.fileno())
                            
                            rec = metrics.get('recall', 'N/A')
                            lat = metrics.get('latency', 'N/A')
                            log(f"    ✓ [{struct_name} | {dataset}] recall: {rec}, latency: {lat}")
                        else:
                            log(f"    ✗ Could not parse metrics from {struct_name} output")
                    else:
                        log(f"    ✗ Execution produced no output or failed")
    finally:
        csv_f.close()
    
    if not dry_run:
        save_results_json(all_results, timestamp)
    
    log(f"\n{'='*80}")
    log("EXPERIMENTS COMPLETE")
    log(f"Total results collected: {len(all_results)}")
    log(f"Results saved to directory: {RESULTS_DIR}")
    log(f"{'='*80}\n")
    
    return all_results


def save_results_json(results, timestamp):
    """Saves structured experiment results into JSON format."""
    if not results:
        return
    json_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"✓ Saved master JSON results to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Automated Experiment Runner for Indexing Structures")
    parser.add_argument("--structures", type=str, help="Comma-separated list of structures (e.g. hnsw,mtree)")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of datasets (e.g. sift,glove)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without running binaries")
    parser.add_argument("--quick", action="store_true", help="Runs single parameter sample test")
    args = parser.parse_args()

    log("Automated Experiment Suite")
    log(f"Workspace Path: {WORKSPACE_ROOT}")
    log(f"Available Datasets: {list(DATASETS.keys())}")
    log(f"Available Structures: {list(PARAM_CONFIGS.keys())}")

    structures = args.structures.split(",") if args.structures else None
    datasets = args.datasets.split(",") if args.datasets else None

    if args.quick:
        log("\n--- QUICK TEST RUN ACTIVATED ---")
        for struct in PARAM_CONFIGS:
            for k in PARAM_CONFIGS[struct]["parameters"]:
                PARAM_CONFIGS[struct]["parameters"][k] = PARAM_CONFIGS[struct]["parameters"][k][:1]

    run_all_experiments(
        structures_to_run=structures, 
        datasets_to_run=datasets, 
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()