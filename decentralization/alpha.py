import pandas as pd
from main import main


def run_alpha_sweep():
    alpha_values = [
        0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0
    ]

    results = []

    for alpha in alpha_values:
        print(f"Running alpha={alpha}")

        result = main(alpha=alpha, plot=False)

        results.append({
            "alpha": alpha,
            "initial_objective": result["initial_objective"],
            "final_objective": result["final_objective"],
            "improvement": result["improvement"],
            "relative_improvement_%": result["relative_improvement"] * 100,
            "iterations": result["iterations"],
        })

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="improvement",
        ascending=False
    )

    df.to_excel("alpha_sweep_with_convergence_results.xlsx", index=False)

    print("\nSaved results to alpha_sweep_results.xlsx")
    print(df)


if __name__ == "__main__":
    run_alpha_sweep()