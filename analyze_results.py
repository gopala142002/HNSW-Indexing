#!/usr/bin/env python3
"""
Results Analysis Tool
Parse, analyze, and visualize experiment results
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_results(csv_file):
    """Load results from CSV file"""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results

def find_best_configs(results, metric="recall", dataset=None):
    """Find best parameter configurations for a metric"""
    
    # Filter by dataset if specified
    filtered = results
    if dataset:
        filtered = [r for r in results if r.get('dataset') == dataset]
    
    # Group by structure and dataset
    by_structure = defaultdict(list)
    for result in filtered:
        structure = result.get('structure', 'unknown')
        by_structure[structure].append(result)
    
    print("\n" + "="*80)
    print(f"BEST CONFIGURATIONS BY {metric.upper()}")
    print("="*80 + "\n")
    
    for structure, configs in sorted(by_structure.items()):
        # Sort by the metric (descending for recall, ascending for time)
        reverse = (metric.lower() in ['recall', 'speedup'])
        
        metric_key = f"metric_{metric}" if f"metric_{metric}" in configs[0] else metric
        
        valid_configs = [c for c in configs if metric_key in c and c[metric_key]]
        
        if not valid_configs:
            print(f"{structure}: No valid data for {metric}")
            continue
        
        # Convert to float for comparison
        try:
            valid_configs.sort(
                key=lambda x: float(x[metric_key]), 
                reverse=reverse
            )
        except (ValueError, TypeError):
            print(f"{structure}: Could not convert {metric} to float")
            continue
        
        # Show top 3
        print(f"{structure} - Top 3 by {metric}:")
        for i, config in enumerate(valid_configs[:3], 1):
            print(f"  {i}. {metric}={config[metric_key]}")
            for k, v in config.items():
                if k.startswith('param_') and v:
                    param_name = k.replace('param_', '')
                    print(f"     {param_name}={v}")
        print()

def compare_structures(results, dataset=None):
    """Compare performance across different structures"""
    
    # Filter by dataset
    filtered = results
    if dataset:
        filtered = [r for r in results if r.get('dataset') == dataset]
    
    # Get unique structures
    structures = set(r.get('structure') for r in filtered if r.get('structure'))
    
    print("\n" + "="*80)
    print(f"PERFORMANCE COMPARISON - {dataset if dataset else 'ALL DATASETS'}")
    print("="*80 + "\n")
    
    for structure in sorted(structures):
        struct_results = [r for r in filtered if r.get('structure') == structure]
        
        if not struct_results:
            continue
        
        print(f"\n{structure}:")
        print("-" * 60)
        
        # Collect metrics
        metrics_data = defaultdict(list)
        for result in struct_results:
            for key, value in result.items():
                if key.startswith('metric_') and value:
                    try:
                        metrics_data[key].append(float(value))
                    except (ValueError, TypeError):
                        pass
        
        # Calculate statistics
        for metric_key, values in sorted(metrics_data.items()):
            metric_name = metric_key.replace('metric_', '')
            if values:
                avg = statistics.mean(values)
                min_v = min(values)
                max_v = max(values)
                print(f"  {metric_name}:")
                print(f"    Avg: {avg:.4f}, Min: {min_v:.4f}, Max: {max_v:.4f}")

def generate_summary_report(csv_file, output_file="summary_report.txt"):
    """Generate a comprehensive summary report"""
    
    results = load_results(csv_file)
    
    if not results:
        print("No results to analyze")
        return
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Group by structure and dataset
        by_struct_dataset = defaultdict(lambda: defaultdict(list))
        for result in results:
            struct = result.get('structure', 'unknown')
            dataset = result.get('dataset', 'unknown')
            by_struct_dataset[struct][dataset].append(result)
        
        # Generate report for each combination
        for structure in sorted(by_struct_dataset.keys()):
            f.write(f"\n{'='*80}\n{structure}\n{'='*80}\n")
            
            for dataset in sorted(by_struct_dataset[structure].keys()):
                configs = by_struct_dataset[structure][dataset]
                f.write(f"\n{dataset.upper()}:\n")
                f.write("-"*60 + "\n")
                f.write(f"Total configurations: {len(configs)}\n\n")
                
                # Summary statistics
                for metric in ['recall', 'build_time', 'latency']:
                    metric_key = f"metric_{metric}"
                    values = []
                    for config in configs:
                        if metric_key in config and config[metric_key]:
                            try:
                                values.append(float(config[metric_key]))
                            except (ValueError, TypeError):
                                pass
                    
                    if values:
                        f.write(f"{metric}:\n")
                        f.write(f"  Mean: {statistics.mean(values):.4f}\n")
                        f.write(f"  Median: {statistics.median(values):.4f}\n")
                        f.write(f"  Min: {min(values):.4f}\n")
                        f.write(f"  Max: {max(values):.4f}\n\n")
    
    print(f"Summary report saved to: {output_file}")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_csv_file> [--dataset DATASET]")
        print("\nExample: python analyze_results.py results_20240806.csv --dataset sift")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    dataset = None
    
    if "--dataset" in sys.argv:
        dataset = sys.argv[sys.argv.index("--dataset") + 1]
    
    if not Path(csv_file).exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    
    results = load_results(csv_file)
    print(f"Loaded {len(results)} results from {csv_file}")
    
    # Generate analyses
    generate_summary_report(csv_file)
    find_best_configs(results, metric="recall", dataset=dataset)
    compare_structures(results, dataset=dataset)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
