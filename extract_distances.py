
import re
import csv
from collections import defaultdict

INPUT_FILE = "combined_test.log"
OUTPUT_FILE = "euclidean_distance_summary_leaf_capacity_32.csv"

single_distances = defaultdict(list)

multi_distances = defaultdict(lambda: {
    "min": [],
    "avg": [],
    "max": []
})

current_multi_approach = None

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as file:
    for line in file:

       
        single_match = re.search(
            r"\[SEED_DISTANCE\]\s+approach=(\S+).*?"
            r"euclidean_l2=([\d.eE+-]+)",
            line
        )

        if single_match:
            approach = single_match.group(1)
            distance = float(single_match.group(2))
            single_distances[approach].append(distance)
            continue

        # Example:
        # [MULTI_SEED_DISTANCE] approach=MTree-Multi number_of_seeds=4
        multi_match = re.search(
            r"\[MULTI_SEED_DISTANCE\]\s+approach=(\S+)",
            line
        )

        if multi_match:
            current_multi_approach = multi_match.group(1)
            continue

        # Example:
        # min_l2=336.73 avg_l2=342.88 max_l2=354.60
        summary_match = re.search(
            r"min_l2=([\d.eE+-]+)\s+"
            r"avg_l2=([\d.eE+-]+)\s+"
            r"max_l2=([\d.eE+-]+)",
            line
        )

        if summary_match and current_multi_approach:
            multi_distances[current_multi_approach]["min"].append(
                float(summary_match.group(1))
            )
            multi_distances[current_multi_approach]["avg"].append(
                float(summary_match.group(2))
            )
            multi_distances[current_multi_approach]["max"].append(
                float(summary_match.group(3))
            )

            current_multi_approach = None


def average(values):
    return sum(values) / len(values) if values else None


# Output one summary row per approach.
approaches = [
    "HNSW-Baseline",
    "MTree",
    "MTree-Multi",
    "PCTree",
    "PCTree-Multi",
    "VPTree",
    "VPTree-Multi",
    "KMeansTree",
    "KMeansTree-Multi"
]

columns = [
    "approach",
    "num_queries",
    "avg_single_euclidean_distance",
    "avg_min_distance",
    "avg_mean_distance",
    "avg_max_distance"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for approach in approaches:
        is_multi = approach.endswith("-Multi")

        if is_multi:
            data = multi_distances[approach]
            counts = [
                len(data["min"]),
                len(data["avg"]),
                len(data["max"])
            ]

            # Avoid silently reporting inconsistent record counts.
            if len(set(counts)) != 1:
                raise ValueError(
                    f"Inconsistent multi-entry record counts for {approach}: "
                    f"{counts}"
                )

            n = counts[0]

            if n == 0:
                print(f"WARNING: No multi-entry records found for {approach}")

            row = {
                "approach": approach,
                "num_queries": n,
                "avg_single_euclidean_distance": "",
                "avg_min_distance": average(data["min"]),
                "avg_mean_distance": average(data["avg"]),
                "avg_max_distance": average(data["max"])
            }

        else:
            distances = single_distances[approach]
            n = len(distances)

            if n == 0:
                print(f"WARNING: No single-entry records found for {approach}")

            row = {
                "approach": approach,
                "num_queries": n,
                "avg_single_euclidean_distance": average(distances),
                "avg_min_distance": "",
                "avg_mean_distance": "",
                "avg_max_distance": ""
            }

        # Format averages to six decimal places.
        for key in columns[2:]:
            if row[key] != "" and row[key] is not None:
                row[key] = f"{row[key]:.6f}"

        writer.writerow(row)

print(f"\nCSV written to: {OUTPUT_FILE}")
print("\nRecord counts:")

for approach in approaches:
    if approach.endswith("-Multi"):
        count = len(multi_distances[approach]["min"])
    else:
        count = len(single_distances[approach])

    print(f"{approach:20s}: {count} queries")
