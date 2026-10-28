import argparse
import csv
import re
import sys
from pathlib import Path

SEED_PATTERN = re.compile(
    r"\[SEED_DISTANCE\]\s+"
    r"approach=(?P<approach>\S+)\s+"
    r".*?\bsquared_l2=(?P<squared_l2>[0-9.eE+-]+)\s+"
    r"euclidean_l2=(?P<euclidean_l2>[0-9.eE+-]+)"
)

APPROACHES = {
    "HNSW-Baseline": "HNSW",
    "PCTree": "PCTree",
    "KMeansTree": "KMeansTree",
    "MTree": "MTree",
    "VPTree": "VPTree",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert [SEED_DISTANCE] records in a benchmark log to CSV."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="fixed_parameter.log",
        help="Input log file (default: fixed_parameter.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="seed_distances.csv",
        help="Output CSV path (default: seed_distances.csv)",
    )
    parser.add_argument(
        "--metric",
        choices=("euclidean_l2", "squared_l2"),
        default="euclidean_l2",
        help="Distance metric to write (default: euclidean_l2)",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_path = Path(args.output)

    if not log_path.is_file():
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        print(
            "Usage: python3 extract_seed_distances.py fixed_parameters.log",
            file=sys.stderr,
        )
        sys.exit(1)

    # Each approach prints one seed-distance record per query. Store records
    # in log order, then align query 1 with query 1 across all approaches.
    distances = {name: [] for name in APPROACHES.values()}
    with log_path.open("r", encoding="utf-8", errors="replace") as log:
        for line in log:
            match = SEED_PATTERN.search(line)
            if not match:
                continue

            approach = match.group("approach")
            if approach not in APPROACHES:
                continue

            column = APPROACHES[approach]
            try:
                value = float(match.group(args.metric))
            except ValueError:
                continue
            distances[column].append(value)

    counts = {name: len(values) for name, values in distances.items()}
    if not any(counts.values()):
        print(
            "Error: no recognized [SEED_DISTANCE] records found. "
            "Check that the log contains approach=... and euclidean_l2=... fields.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("Records found per approach:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    # Avoid silently pairing unrelated queries if an approach is missing
    # records. Write rows up to the shortest available series and warn.
    nonzero_counts = [count for count in counts.values() if count > 0]
    if len(nonzero_counts) != len(counts) or len(set(counts.values())) != 1:
        print(
            "Warning: approach record counts differ. The CSV will contain "
            "rows only up to the shortest non-empty series; missing values "
            "will be left blank.",
            file=sys.stderr,
        )

    row_count = max(counts.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["query_id", "HNSW_distance", "PCTree_distance",
               "KMeansTree_distance", "MTree_distance", "VPTree_distance"]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for index in range(row_count):
            row = {"query_id": index + 1}
            for name in distances:
                row[f"{name}_distance"] = (
                    distances[name][index] if index < len(distances[name]) else ""
                )
            writer.writerow(row)

    print(f"\nWrote {row_count} query rows to: {output_path.resolve()}")
    print(f"Distance metric: {args.metric}")

    # Print counts out of 10,000 (or the actual number of paired records)
    # where each tree's seed is closer to the query than the HNSW seed.
    baseline = distances["HNSW"]
    print("\\nQueries where tree seed distance is lower than HNSW:")
    for name in ("PCTree", "KMeansTree", "MTree", "VPTree"):
        tree_values = distances[name]
        paired_count = min(len(baseline), len(tree_values))
        lower = sum(
            tree_values[i] < baseline[i]
            for i in range(paired_count)
        )
        print(f"  {name}: {lower} out of {paired_count} queries")



if __name__ == "__main__":
    main()