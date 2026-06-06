"""
Experiment: effect of permanently offline nodes on profile steering performance.

A fixed subset of nodes is chosen once at the start of each run and is completely
absent for the entire experiment — they never aggregate, never steer, and their
local_profile stays at the initialised (unoptimised) value.

This differs from the other two experiments:
  - non_steering:    nodes aggregate normally but skip profile_steering_step
  - non_aggregating: nodes skip gossip per-iteration but steer; subset re-drawn each round
  - permanently_offline (this): nodes are gone entirely — no gossip, no steering, fixed load

Online nodes use population = N_total, so when offline_fraction > 0 their push-sum
estimates are biased upward by N_total / N_online. The true objective is computed
over all N nodes, so the offline nodes' unoptimised load counts against the system.

Sweep: offline_fraction × seed × season  (6 × 5 × 4 = 120 runs)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from main_permanently_offline import main
from main_real_data import HOUSE_IDS

ALPHA = 0.075

DAYS = {
    "winter": "2023-01-15",
    "spring": "2023-05-15",
    "summer": "2023-08-15",
    "autumn": "2023-10-15",
}

CRASH_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [42, 137, 271, 404, 531]


def run_experiment():
    records = []

    total = len(CRASH_FRACTIONS) * len(SEEDS) * len(DAYS)
    done = 0

    for fraction in CRASH_FRACTIONS:
        n_offline = round(fraction * len(HOUSE_IDS))
        print(f"\n{'='*65}")
        print(f"Offline fraction: {fraction:.0%}  ({n_offline}/{len(HOUSE_IDS)} nodes permanently absent)")

        for seed in SEEDS:
            for season, day in DAYS.items():
                done += 1
                print(f"  [{done}/{total}] seed={seed}  season={season} ({day})")
                result = main(
                    day=day,
                    alpha=ALPHA,
                    offline_fraction=fraction,
                    seed=seed,
                    plot=False,
                )

                records.append({
                    "offline_fraction": result["offline_fraction"],
                    "offline_count": result["offline_count"],
                    "online_count": result["online_count"],
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
        df.groupby("offline_fraction")
        .agg(
            offline_count=("offline_count", "first"),
            online_count=("online_count", "first"),
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
    raw_path = os.path.join(out_dir, "permanently_offline_raw.xlsx")
    summary_path = os.path.join(out_dir, "permanently_offline_summary.xlsx")

    df.to_excel(raw_path, index=False)
    summary.to_excel(summary_path, index=False)

    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")

    return df, summary


if __name__ == "__main__":
    run_experiment()
