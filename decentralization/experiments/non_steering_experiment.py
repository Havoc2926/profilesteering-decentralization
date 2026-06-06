"""
Experiment: effect of non-steering node proportion on profile steering performance.

For each proportion level the same fixed subset of houses is held out from steering
across every run, isolating the effect of participation rate from noise due to
which specific nodes are non-steering.

Runs are repeated across several representative days (one per season) to capture
seasonal variation in load and PV generation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from main_real_data import main, HOUSE_IDS

# One representative day per season
DAYS = {
    "winter":  "2023-01-15",
    "spring":  "2023-05-15",
    "summer":  "2023-08-15",
    "autumn":  "2023-10-15",
}

ALPHA = 0.075

# Proportions to test; counts are rounded to nearest whole house
PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def fixed_non_steering_ids(proportion: float) -> list[int]:
    """
    Deterministically pick which houses are non-steering for a given proportion.
    Uses the first N house IDs (sorted) so the subset is always the same and
    strictly nested as proportion grows.
    """
    n = round(proportion * len(HOUSE_IDS))
    return sorted(HOUSE_IDS)[:n]


def run_experiment():
    records = []

    for proportion in PROPORTIONS:
        ns_ids = fixed_non_steering_ids(proportion)
        n_ns = len(ns_ids)
        print(f"\n{'='*60}")
        print(f"Non-steering proportion: {proportion:.0%}  ({n_ns}/{len(HOUSE_IDS)} houses)")
        print(f"Non-steering house IDs: {ns_ids}")

        for season, day in DAYS.items():
            print(f"  Season: {season} ({day})")
            result = main(
                day=day,
                alpha=ALPHA,
                non_steering_ids=ns_ids,
                plot=False,
            )

            records.append({
                "non_steering_proportion": proportion,
                "non_steering_count": n_ns,
                "non_steering_ids": str(ns_ids),
                "season": season,
                "day": day,
                "alpha": ALPHA,
                "initial_objective": result["initial_objective"],
                "final_objective": result["final_objective"],
                "improvement": result["improvement"],
                "relative_improvement_%": result["relative_improvement"] * 100,
                "iterations": result["iterations"],
                "aggregate_rmse": result["aggregate_rmse"],
            })

    df = pd.DataFrame(records)

    summary = (
        df.groupby("non_steering_proportion")
        .agg(
            non_steering_count=("non_steering_count", "first"),
            mean_initial_objective=("initial_objective", "mean"),
            mean_final_objective=("final_objective", "mean"),
            std_final_objective=("final_objective", "std"),
            mean_relative_improvement=("relative_improvement_%", "mean"),
            std_relative_improvement=("relative_improvement_%", "std"),
            mean_iterations=("iterations", "mean"),
            mean_aggregate_rmse=("aggregate_rmse", "mean"),
        )
        .reset_index()
    )

    print("\n\n=== Summary ===")
    print(summary.to_string(index=False))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(out_dir, "non_steering_raw.xlsx")
    summary_path = os.path.join(out_dir, "non_steering_summary.xlsx")

    df.to_excel(raw_path, index=False)
    summary.to_excel(summary_path, index=False)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")

    return df, summary


if __name__ == "__main__":
    run_experiment()
