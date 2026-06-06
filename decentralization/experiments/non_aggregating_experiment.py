"""
Experiment: effect of non-aggregating nodes on profile steering performance.

Each PS iteration, a random subset of nodes is taken offline at the start of the
gossip phase and rejoins after a configurable number of rounds (offline_duration).
The full ceil(N*ln(N)) gossip rounds always run; only the absent nodes' duration
varies. The subset is re-drawn every iteration so no single node is permanently excluded.

The sweep covers:
  - non_aggregating_proportion: fraction of nodes offline at the start of each gossip phase
  - offline_duration_rounds: how many gossip rounds they stay offline before rejoining
    (None = entire gossip phase, i.e. worst case — no contribution at all)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import math
import pandas as pd
from main_real_data import main, HOUSE_IDS

ALPHA = 0.075

# One representative day per season for seasonal variation
DAYS = {
    "winter": "2023-01-15",
    "spring": "2023-05-15",
    "summer": "2023-08-15",
    "autumn": "2023-10-15",
}

PROPORTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 137, 271, 404, 531]

# Total gossip always runs for ceil(N * ln(N)) ≈ 77 rounds.
# non_aggregating_duration controls how many of those rounds the absent nodes stay offline.
# None means offline for the full gossip phase (worst case).
DEFAULT_GOSSIP = math.ceil(len(HOUSE_IDS) * math.log(len(HOUSE_IDS)))
OFFLINE_DURATIONS = [10, 20, 40, DEFAULT_GOSSIP, None]  # None = full phase


def run_experiment():
    records = []

    total = len(PROPORTIONS) * len(OFFLINE_DURATIONS) * len(SEEDS) * len(DAYS)
    done = 0

    for offline_duration in OFFLINE_DURATIONS:
        duration_label = offline_duration if offline_duration is not None else f"{DEFAULT_GOSSIP} (full)"
        for proportion in PROPORTIONS:
            n_absent = round(proportion * len(HOUSE_IDS))
            print(f"\n{'='*65}")
            print(f"offline_duration={duration_label}  |  non-aggregating proportion={proportion:.0%}"
                  f"  ({n_absent}/{len(HOUSE_IDS)} nodes absent per iteration)")

            for seed in SEEDS:
                for season, day in DAYS.items():
                    done += 1
                    print(f"  [{done}/{total}] seed={seed}  season={season} ({day})")
                    result = main(
                        day=day,
                        alpha=ALPHA,
                        non_aggregating_proportion=proportion,
                        non_aggregating_duration=offline_duration,
                        seed=seed,
                        plot=False,
                    )

                    records.append({
                        "offline_duration_rounds": result["non_aggregating_duration"],
                        "total_gossip_rounds": result["gossip_rounds"],
                        "non_aggregating_proportion": proportion,
                        "non_aggregating_count": n_absent,
                        "seed": seed,
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
        df.groupby(["offline_duration_rounds", "non_aggregating_proportion"])
        .agg(
            non_aggregating_count=("non_aggregating_count", "first"),
            n_runs=("final_objective", "count"),
            mean_initial_objective=("initial_objective", "mean"),
            mean_final_objective=("final_objective", "mean"),
            std_final_objective=("final_objective", "std"),
            mean_relative_improvement=("relative_improvement_%", "mean"),
            std_relative_improvement=("relative_improvement_%", "std"),
            mean_iterations=("iterations", "mean"),
            mean_aggregate_rmse=("aggregate_rmse", "mean"),
            std_aggregate_rmse=("aggregate_rmse", "std"),
        )
        .reset_index()
    )

    print("\n\n=== Summary ===")
    print(summary.to_string(index=False))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(out_dir, "non_aggregating_raw.xlsx")
    summary_path = os.path.join(out_dir, "non_aggregating_summary.xlsx")

    df.to_excel(raw_path, index=False)
    summary.to_excel(summary_path, index=False)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")

    return df, summary


if __name__ == "__main__":
    run_experiment()
