import pandas as pd
from main import main


def run_participation_sweep(alpha=0.075, runs_per_proportion=5):
    # Proportions of nodes that do NOT participate in profile steering
    proportions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []

    for proportion in proportions:
        print(f"\n=== Non-steering proportion: {proportion:.0%} ===")

        for run in range(runs_per_proportion):
            print(f"  Run {run + 1}/{runs_per_proportion}")
            result = main(alpha=alpha, non_steering_proportion=proportion, plot=False)

            results.append({
                "non_steering_proportion": proportion,
                "non_steering_count": result["non_steering_count"],
                "run": run + 1,
                "alpha": alpha,
                "initial_objective": result["initial_objective"],
                "final_objective": result["final_objective"],
                "improvement": result["improvement"],
                "relative_improvement_%": result["relative_improvement"] * 100,
                "iterations": result["iterations"],
            })

    df = pd.DataFrame(results)

    summary = df.groupby("non_steering_proportion").agg(
        mean_final_objective=("final_objective", "mean"),
        std_final_objective=("final_objective", "std"),
        mean_improvement=("improvement", "mean"),
        mean_relative_improvement=("relative_improvement_%", "mean"),
        mean_iterations=("iterations", "mean"),
    ).reset_index()

    print("\n=== Summary ===")
    print(summary.to_string())

    df.to_excel("participation_sweep_raw.xlsx", index=False)
    summary.to_excel("participation_sweep_summary.xlsx", index=False)

    print("\nSaved results to participation_sweep_raw.xlsx and participation_sweep_summary.xlsx")


if __name__ == "__main__":
    run_participation_sweep(alpha=0.075, runs_per_proportion=5)
