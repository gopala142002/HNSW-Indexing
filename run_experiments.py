import os
import subprocess
import csv
import json
import re
from pathlib import Path
from datetime import datetime
import sys
import itertools
from tqdm import tqdm



WORKSPACE_ROOT = Path(__file__).parent
BUILD_DIR = WORKSPACE_ROOT / "build"
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
RESULTS_DIR = WORKSPACE_ROOT / "results_experiments"


RESULTS_DIR.mkdir(exist_ok=True)

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
    "sift": "./datasets/sift",
    "glove": "./datasets/glove",
    "gist": "./datasets/gist"
}

def log(msg, end="\n"):
    print(f"{msg}", end=end)

def build_project():
    log("Checking for compiled binaries...")
    
    required_binaries = ["test_hnsw", "test_mtree", "test_vpttree", "test_pctree", "test_vtree"]
    
    log("Building binaries with g++...")
    try:
        compile_commands = [
            "g++ -O3 -std=c++17 -I. test_hnsw.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o test_hnsw",
            "g++ -O3 -std=c++17 -I. test_mtree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o test_mtree",
            "g++ -O3 -std=c++17 -I. test_vpttree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o test_vpttree",
            "g++ -O3 -std=c++17 -I. test_pctree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o test_pctree",
            "g++ -O3 -std=c++17 -I. test_vtree.cpp hnswlib/pctree.cpp hnswlib/vtree.cpp hnswlib/mtree.cpp hnswlib/vpttree.cpp -o test_vtree",
        ]
        
        for cmd in compile_commands:
            log(f"  Compiling: {cmd.split()[2]}")
            subprocess.run(
                cmd,
                shell=True,
                cwd=WORKSPACE_ROOT,
                check=True,
                capture_output=True
            )
        
        log("✓ Binaries Building successful!")
        return True
    except subprocess.CalledProcessError as e:
        log(f"✗ Build failed: {e}")
        return False

def run_test(binary_path, dataset_path, params_list, test_name):
    try:
        cmd = [str(binary_path), str(dataset_path)] + [str(p) for p in params_list]
        
        log(f"  Running: {' '.join([Path(str(cmd[0])).name] + cmd[1:])}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        return result.stdout
    except Exception as e:
        log(f"  ✗ ERROR running test: {e}")
        return None

def parse_output(output):
    if not output:
        return None
    
    metrics = {}
    
    patterns = {
        'recall': r'Recall[:\s@]+\d+\s*=\s*([0-9.]+)',
        'build_time': r'built in\s+([0-9.]+)\s*s',
        'latency': r'latency[:\s]+([0-9.]+)\s*u?s',
        'height': r'height[:\s]+(\d+)',
        'speedup': r'Speedup[:\s/()]+([0-9.]+)',
    }
    
    for metric_name, pattern in patterns.items():
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                # Use the last match (usually the final result)
                metrics[metric_name] = float(matches[-1])
            except ValueError:
                pass
    
    return metrics if metrics else None

def generate_parameter_combinations(params_dict):
    """Generate all combinations of parameters"""
    import itertools
    
    param_names = list(params_dict.keys())
    param_values = [params_dict[name] for name in param_names]
    
    combinations = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)
    
    return combinations

def run_all_experiments(structures_to_run=None, datasets_to_run=None):
    log("STARTING COMPREHENSIVE EXPERIMENT SUITE")
    
    if structures_to_run is None:
        structures_to_run = list(PARAM_CONFIGS.keys())
    else:
        structures_to_run = [s for s in structures_to_run if s in PARAM_CONFIGS]
    
    if datasets_to_run is None:
        datasets_to_run = list(DATASETS.keys())
    else:
        datasets_to_run = [d for d in datasets_to_run if d in DATASETS]
    
    all_results = []
    
    if not build_project():
        log("Build failed. Aborting.")
        return all_results
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    csv_f = open(csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
    csv_writer.writeheader()
    csv_f.flush()
    
    log(f"Writing incremental CSV to: {csv_file}")
    
    for struct_name in tqdm(structures_to_run, desc="Structures", colour="green"):
        struct_config = PARAM_CONFIGS[struct_name]
        
        log(f"Testing {struct_name.upper()}")
        
        param_combos = generate_parameter_combinations(struct_config["parameters"])
        log(f"Parameter combinations to test: {len(param_combos)}")
        
        for combo_idx, param_combo in enumerate(tqdm(param_combos, desc=f"  {struct_name} configs", leave=False), 1):
            log(f"\n[{struct_name}] Config {combo_idx}/{len(param_combos)}: ", end="")
            log(f"{param_combo}")
            
            arg_order = struct_config["arg_order"]
            param_args = [param_combo[param_name] for param_name in arg_order]
            
            for dataset in tqdm(datasets_to_run, desc=f"    datasets", leave=False):
                dataset_path = DATASETS_DIR / dataset
                
                if not dataset_path.exists():
                    log(f"  ✗ Dataset not found: {dataset_path}")
                    continue
                
                log(f"\n  Dataset: {dataset}")
                
                binary_path = BUILD_DIR / struct_config["binary"]
                if not binary_path.exists():
                    binary_path = WORKSPACE_ROOT / struct_config["binary"]
                
                if not binary_path.exists():
                    log(f"  ✗ Binary not found: {binary_path}")
                    continue
                
                test_name = f"{struct_name}_{combo_idx}"
                output = run_test(binary_path, dataset_path, param_args, test_name)
                
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
                        
                        recall_value = metrics.get('recall')
                        latency_value = metrics.get('latency')
                        recall_str = f"{recall_value:.4f}" if isinstance(recall_value, float) else str(recall_value)
                        latency_str = f"{latency_value:.1f} us" if isinstance(latency_value, float) else str(latency_value)
                        log(f"    ✓ Results: recall={recall_str}, latency={latency_str}")
                    else:
                        log(f"    ✗ Could not parse output")
                else:
                    log(f"    ✗ Test execution failed")
    
    save_results(all_results)
    
    log(f"\n{'='*80}")
    log(f"EXPERIMENTS COMPLETE")
    log(f"Total results collected: {len(all_results)}")
    log(f"Results saved to: {RESULTS_DIR}")
    log(f"{'='*80}\n")
    
    return all_results

def save_results(results):
    if not results:
        log("No results to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"✓ Saved JSON results to: {json_file}")
    
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    
    flat_results = []
    for result in results:
        row = {
            "structure": result["structure"],
            "dataset": result["dataset"],
        }
    
        for param_name, param_value in result["parameters"].items():
            row[f"param_{param_name}"] = param_value
        
        for metric_name, metric_value in result["metrics"].items():
            row[f"metric_{metric_name}"] = metric_value
        
        flat_results.append(row)
    
    if flat_results:
        fieldnames = flat_results[0].keys()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_results)
        log(f"✓ Saved CSV results to: {csv_file}")

def main():
    log("Automated Experiment Runner for HNSW/Trees Comparison")
    log(f"Workspace: {WORKSPACE_ROOT}")
    log(f"Datasets: {list(DATASETS.keys())}")
    log(f"Structures: {list(PARAM_CONFIGS.keys())}")
    
    for dataset, path in DATASETS.items():
        dataset_path = DATASETS_DIR / dataset
        if not dataset_path.exists():
            log(f"WARNING: Dataset not found: {dataset_path}")
    
    structures = None
    datasets = None
    
    if len(sys.argv) > 1:
        if "--structures" in sys.argv:
            idx = sys.argv.index("--structures")
            structures = sys.argv[idx + 1].split(",")
        
        if "--datasets" in sys.argv:
            idx = sys.argv.index("--datasets")
            datasets = sys.argv[idx + 1].split(",")
    
    run_all_experiments(structures_to_run=structures, datasets_to_run=datasets)

if __name__ == "__main__":
    main()
