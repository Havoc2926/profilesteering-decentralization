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
from main_real_data import (
    RealLoad,
    load_day_profiles,
    date_to_day_index,
    add_profiles,
    objective,
    INTERVALS_PER_DAY,
    HOUSE_IDS,
    DATA_PATH,
)


def compute_true_aggregate(nodes, profile_length):
    total = [0.0] * profile_length
    for node in nodes:
        total = add_profiles(total, node.local_profile)
    return total


def _assign_random_topology(nodes, edge_prob=0.3):
    """Random connected graph (Erdos-Renyi + spanning tree) over the given node list."""
    n = len(nodes)
    if n == 0:
        return
    adj = {i: set() for i in range(n)}

    unvisited = list(range(1, n))
    random.shuffle(unvisited)
    visited = [0]
    for v in unvisited:
        u = random.choice(visited)
        adj[u].add(v)
        adj[v].add(u)
        visited.append(v)

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
    offline_fraction=0.0,
    offline_ids=None,
    seed=None,
    plot=True,
    with_battery=True,
):
    """
    Run profile steering where a fixed subset of nodes is permanently offline
    for the entire experiment — they never aggregate, never steer, and their
    local_profile stays at the initialised value throughout.

    Online nodes use population = total N, so their push-sum estimates are
    biased upward by N_total / N_online when offline_fraction > 0.
    The true objective is computed over all N nodes (offline nodes' static
    consumption counts against the system target).

    Parameters
    ----------
    offline_fraction : float
        Fraction of nodes to take permanently offline. Ignored when offline_ids
        is given explicitly.
    offline_ids : list[int] | None
        Explicit list of house IDs to take offline. Overrides offline_fraction.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    day_index = date_to_day_index(day) if isinstance(day, str) else int(day)
    print(f"Loading day index {day_index} ({day})")

    house_profiles = load_day_profiles(day_index)
    population = len(HOUSE_IDS)
    gossip_rounds = math.ceil(population * math.log(population))
    ps_rounds = 100

    initial_aggregate = [0.0] * INTERVALS_PER_DAY
    for profile in house_profiles.values():
        initial_aggregate = add_profiles(initial_aggregate, profile)
    mean_power = sum(initial_aggregate) / INTERVALS_PER_DAY
    p = [mean_power] * INTERVALS_PER_DAY

    # Build all nodes (offline ones get init_profile but never participate)
    all_nodes = []
    for house_id in HOUSE_IDS:
        devices = [RealLoad(house_profiles[house_id])]
        if with_battery:
            devices.append(Battery())
        # population = total N so online nodes' estimates are scaled correctly
        # relative to the full system target
        node = Node(id=house_id, devices=devices, desired_profile=p, population=population)
        all_nodes.append(node)

    # Choose offline set once — fixed for the entire run
    if offline_ids is not None:
        offline_id_set = set(offline_ids)
    else:
        n_offline = round(offline_fraction * population)
        offline_id_set = set(random.sample(HOUSE_IDS, n_offline))

    offline_nodes = [n for n in all_nodes if n.id in offline_id_set]
    online_nodes  = [n for n in all_nodes if n.id not in offline_id_set]

    print(f"Offline nodes ({len(offline_nodes)}/{population}): {sorted(offline_id_set)}")
    print(f"Online nodes  ({len(online_nodes)}/{population})")

    # Topology is built only over online nodes; offline nodes have no neighbours
    _assign_random_topology(online_nodes)

    for node in all_nodes:
        node.init_profile()

    initial_x = compute_true_aggregate(all_nodes, INTERVALS_PER_DAY)
    initial_obj = objective(initial_x, p)

    objective_history = []
    rmse_history = []

    print(f"Initial objective: {initial_obj:.2f}  (target = flat {mean_power:.1f} W)")
    previous_obj = initial_obj

    for ps_iter in range(ps_rounds):
        print(f"\nProfile Steering iteration {ps_iter}")

        # Only online nodes participate in push-sum
        for node in online_nodes:
            node.reset_push_sum()

        for _ in range(gossip_rounds):
            for node in online_nodes:
                node.send()
            for node in online_nodes:
                node.receive()

        # RMSE: compare each online node's estimate against the true aggregate
        # of ALL nodes (including the offline nodes' static load)
        true_x_before_steering = compute_true_aggregate(all_nodes, INTERVALS_PER_DAY)
        node_rmses = []
        for node in online_nodes:
            estimate = node.make_estimate()
            rmse = np.sqrt(np.mean([(e - t) ** 2 for e, t in zip(estimate, true_x_before_steering)]))
            node_rmses.append(rmse)
            error = np.linalg.norm([e - t for e, t in zip(estimate, true_x_before_steering)])
            print(f"  Node {node.id} estimate error: {error:.4f}")
        rmse_history.append(float(np.mean(node_rmses)) if node_rmses else 0.0)

        improvements = []
        for node in online_nodes:
            improvement = node.profile_steering_step(alpha)
            improvements.append(improvement)

        # Objective is over all nodes — offline ones contribute their fixed load
        true_x = compute_true_aggregate(all_nodes, INTERVALS_PER_DAY)
        obj = objective(true_x, p)
        objective_history.append(obj)

        print(f"  Objective: {obj:.2f}  |  Total improvement: {sum(improvements):.4f}")

        if abs(previous_obj - obj) < 0.001:
            print("Converged")
            break

        previous_obj = obj

    print("\nFinished!")
    final_x = compute_true_aggregate(all_nodes, INTERVALS_PER_DAY)
    final_obj = objective(final_x, p)
    aggregate_rmse = float(np.mean(rmse_history)) if rmse_history else 0.0
    print(f"Initial objective : {initial_obj:.2f}")
    print(f"Final objective   : {final_obj:.2f}")
    print(f"Improvement       : {initial_obj - final_obj:.2f}")
    print(f"Aggregate RMSE    : {aggregate_rmse:.4f}")

    time_axis = [i * 15 for i in range(INTERVALS_PER_DAY)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Permanently offline — {day} (day {day_index})  |  offline={len(offline_nodes)}/{population}")

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
        "offline_fraction": len(offline_nodes) / population,
        "offline_count": len(offline_nodes),
        "online_count": len(online_nodes),
        "offline_ids": sorted(offline_id_set),
        "gossip_rounds": gossip_rounds,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement": initial_obj - final_obj,
        "relative_improvement": (initial_obj - final_obj) / initial_obj,
        "iterations": len(objective_history),
        "aggregate_rmse": aggregate_rmse,
    }


if __name__ == "__main__":
    main(day="2023-08-15", alpha=0.075, offline_fraction=0.2, plot=True)
