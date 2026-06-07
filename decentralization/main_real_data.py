import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from PushSum import Node
from dev.battery import Battery

INTERVALS_PER_DAY = 96
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_aardehuizen_without_carport.csv")
HOUSE_IDS = list(range(24))  # houses 0-23


class RealLoad:
    """Inflexible baseload device seeded from real measurement data (load + PV)."""

    def __init__(self, fixed_profile):
        self.profile = list(fixed_profile)
        self.candidate = []

    def init(self, p):
        return list(self.profile)

    def plan(self, d):
        self.candidate = list(self.profile)
        return 0.0

    def accept(self):
        return [0.0] * len(self.profile)


def load_day_profiles(day_index: int) -> dict[int, list[float]]:
    """Return {house_id: net_power_profile} for the given day (0-indexed)."""
    df = pd.read_csv(DATA_PATH)
    start = day_index * INTERVALS_PER_DAY
    end = start + INTERVALS_PER_DAY
    day_df = df.iloc[start:end].reset_index(drop=True)

    profiles = {}
    for house_id in HOUSE_IDS:
        load_col = f"load_house_{house_id}_W"
        pv_col = f"pv_house_{house_id}_W"

        load = day_df[load_col].tolist() if load_col in day_df.columns else [0.0] * INTERVALS_PER_DAY
        pv = day_df[pv_col].tolist() if pv_col in day_df.columns else [0.0] * INTERVALS_PER_DAY

        profiles[house_id] = [l + p for l, p in zip(load, pv)]

    return profiles


def date_to_day_index(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to a 0-based day index within the 2023 dataset."""
    from datetime import date
    d = date.fromisoformat(date_str)
    return (d - date(2023, 1, 1)).days


def add_profiles(a, b):
    return [x + y for x, y in zip(a, b)]


def compute_true_aggregate(nodes, profile_length):
    total = [0.0] * profile_length
    for node in nodes:
        total = add_profiles(total, node.local_profile)
    return total


def objective(x, p):
    return np.linalg.norm([x_i - p_i for x_i, p_i in zip(x, p)])


def _assign_random_topology(nodes, edge_prob=0.3):
    """
    Build a random connected graph over nodes using Erdos-Renyi with the given
    edge probability, then add a random spanning tree to guarantee connectivity.
    Each node's neighbour list is set (excluding itself, undirected).
    """
    n = len(nodes)
    adj = {i: set() for i in range(n)}

    # Random spanning tree via a random walk (guarantees connectivity)
    unvisited = list(range(1, n))
    random.shuffle(unvisited)
    visited = [0]
    for v in unvisited:
        u = random.choice(visited)
        adj[u].add(v)
        adj[v].add(u)
        visited.append(v)

    # Add extra random edges
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i] and random.random() < edge_prob:
                adj[i].add(j)
                adj[j].add(i)

    for i, node in enumerate(nodes):
        node.add_neighbours([nodes[j] for j in sorted(adj[i])])


def main(
    day="2023-07-15",
    alpha=0.075,
    non_steering_proportion=0.0,
    non_steering_ids=None,
    crash_fraction=0.0,
    crash_duration_rounds=0,
    non_aggregating_proportion=0.0,
    non_aggregating_duration=None,
    seed=None,
    plot=True,
    with_battery=True,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    day_index = date_to_day_index(day) if isinstance(day, str) else int(day)
    print(f"Loading day index {day_index} ({day})")

    house_profiles = load_day_profiles(day_index)
    population = len(HOUSE_IDS)
    gossip_rounds = math.ceil(population * math.log(population))
    ps_rounds = 100

    # Use a flat target equal to the mean aggregate power (minimise peaks)
    initial_aggregate = [0.0] * INTERVALS_PER_DAY
    for profile in house_profiles.values():
        initial_aggregate = add_profiles(initial_aggregate, profile)
    mean_power = sum(initial_aggregate) / INTERVALS_PER_DAY
    p = [mean_power] * INTERVALS_PER_DAY

    nodes = []
    for house_id in HOUSE_IDS:
        devices = [RealLoad(house_profiles[house_id])]
        if with_battery:
            devices.append(Battery())
        node = Node(id=house_id, devices=devices, desired_profile=p, population=population)
        nodes.append(node)

    if non_steering_ids is not None:
        non_steering_ids = set(non_steering_ids)
    else:
        num_non_steering = int(non_steering_proportion * population)
        non_steering_ids = set(random.sample(range(population), num_non_steering))
        print(f"Non-steering nodes ({len(non_steering_ids)}/{population}): {sorted(non_steering_ids)}")

    num_non_aggregating = int(non_aggregating_proportion * population)

    _assign_random_topology(nodes)

    for node in nodes:
        node.init_profile()

    initial_x = compute_true_aggregate(nodes, INTERVALS_PER_DAY)
    initial_obj = objective(initial_x, p)

    objective_history = []
    rmse_history = []
    num_crashed = int(crash_fraction * population)

    print(f"Initial objective: {initial_obj:.2f}  (target = flat {mean_power:.1f} W)")
    previous_obj = initial_obj

    for ps_iter in range(ps_rounds):
        print(f"\nProfile Steering iteration {ps_iter}")

        for node in nodes:
            node.reset_push_sum()

        crashed_nodes = random.sample(nodes, num_crashed) if num_crashed > 0 else []
        for node in crashed_nodes:
            node.crash()

        # Non-aggregating nodes are offline for non_aggregating_duration rounds then rejoin;
        # subset is re-drawn each iteration. Full gossip_rounds always run.
        non_aggregating_nodes = random.sample(nodes, num_non_aggregating) if num_non_aggregating > 0 else []
        offline_until = non_aggregating_duration if non_aggregating_duration is not None else gossip_rounds
        for node in non_aggregating_nodes:
            node.crash()

        for gossip_iter in range(gossip_rounds):
            if gossip_iter == crash_duration_rounds:
                for node in crashed_nodes:
                    node.recover()
            if gossip_iter == offline_until:
                for node in non_aggregating_nodes:
                    node.recover()
            for node in nodes:
                node.send()
            for node in nodes:
                node.receive()

        for node in nodes:
            node.recover()

        true_x_before_steering = compute_true_aggregate(nodes, INTERVALS_PER_DAY)
        node_rmses = []
        for node in nodes:
            estimate = node.make_estimate()
            rmse = np.sqrt(np.mean([(e - t) ** 2 for e, t in zip(estimate, true_x_before_steering)]))
            node_rmses.append(rmse)
            error = np.linalg.norm([e - t for e, t in zip(estimate, true_x_before_steering)])
            print(f"  Node {node.id} estimate error: {error:.4f}")
        rmse_history.append(float(np.mean(node_rmses)))

        improvements = []
        for node in nodes:
            if node.id in non_steering_ids:
                continue
            improvement = node.profile_steering_step(alpha)
            improvements.append(improvement)

        true_x = compute_true_aggregate(nodes, INTERVALS_PER_DAY)
        obj = objective(true_x, p)
        objective_history.append(obj)

        print(f"  Objective: {obj:.2f}  |  Total improvement: {sum(improvements):.4f}")

        if abs(previous_obj - obj) < 0.001:
            print("Converged")
            break

        previous_obj = obj

    print("\nFinished!")
    final_x = compute_true_aggregate(nodes, INTERVALS_PER_DAY)
    final_obj = objective(final_x, p)
    aggregate_rmse = float(np.mean(rmse_history))
    print(f"Initial objective : {initial_obj:.2f}")
    print(f"Final objective   : {final_obj:.2f}")
    print(f"Improvement       : {initial_obj - final_obj:.2f}")
    print(f"Aggregate RMSE    : {aggregate_rmse:.4f}")

    time_axis = [i * 15 for i in range(INTERVALS_PER_DAY)]  # minutes from midnight

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Real-data run — {day} (day {day_index})")

    axes[0].plot(range(len(objective_history)), objective_history, marker="o")
    axes[0].axhline(initial_obj, linestyle="--", label="Initial objective")
    axes[0].set_xlabel("PS iteration")
    axes[0].set_ylabel("Objective (L2 norm)")
    axes[0].set_title("Convergence")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_axis, initial_x, label="Before steering", alpha=0.7)
    axes[1].plot(time_axis, final_x, label="After steering", alpha=0.7)
    axes[1].axhline(mean_power, linestyle="--", color="gray", label=f"Target ({mean_power:.0f} W)")
    axes[1].set_xlabel("Time of day (minutes)")
    axes[1].set_ylabel("Aggregate power (W)")
    axes[1].set_title("Aggregate profile")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if plot:
        plt.show()

    return {
        "day": day,
        "day_index": day_index,
        "alpha": alpha,
        "non_steering_proportion": non_steering_proportion,
        "non_aggregating_proportion": non_aggregating_proportion,
        "non_aggregating_duration": offline_until,
        "crash_fraction": crash_fraction,
        "crash_duration_rounds": crash_duration_rounds,
        "gossip_rounds": gossip_rounds,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement": initial_obj - final_obj,
        "relative_improvement": (initial_obj - final_obj) / initial_obj,
        "iterations": len(objective_history),
        "aggregate_rmse": aggregate_rmse,
    }


if __name__ == "__main__":
    main(day="2023-07-15", alpha=0.075, with_battery=True, plot=True)
