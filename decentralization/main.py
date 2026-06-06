from PushSum import Node
import random
import matplotlib.pyplot as plt
import numpy as np
from dev.load import Load
from dev.battery import Battery
from dev.heatpump import HeatPump
from dev.electricvehicle import ElectricVehicle


# Helper functions
def add_profiles(a, b):
    return [x + y for x, y in zip(a, b)]

def compute_true_aggregate(nodes, profile_length):
    total = [0.0] * profile_length

    for node in nodes:
        total = add_profiles(total, node.local_profile)

    return total

def objective(x, p):
    return np.linalg.norm([x_i - p_i for x_i, p_i in zip(x, p)])

def create_devices():
    return [
        Load(),
        Battery(),
        ElectricVehicle(),
        HeatPump(),
    ]

def main(alpha=1.0, non_steering_proportion=0.0,
         crash_fraction=0.0, crash_duration_rounds=0,
         seed=None, plot=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    profile_length = 96
    population = 10
    gossip_rounds = 60
    ps_rounds = 100

    #Desired global profile
    p= [0.0] * profile_length
    nodes = []

    for i in range(population):
        devices = create_devices()

        node = Node(id=i, devices=devices, desired_profile=p, population=population)
        nodes.append(node)

    num_non_steering = int(non_steering_proportion * population)
    non_steering_ids = set(random.sample(range(population), num_non_steering))
    print(f"Non-steering nodes ({num_non_steering}/{population}): {sorted(non_steering_ids)}")

    for i, node in enumerate(nodes):
        neighbours = [
            nodes[(i - 1) % population],
            nodes[(i + 1) % population]
        ]

        node.add_neighbours(neighbours)

    for node in nodes:
        node.init_profile()

    initial_x = compute_true_aggregate(nodes, profile_length)
    initial_obj = objective(initial_x, p)

    objective_history = []
    rmse_history = []

    num_crashed = int(crash_fraction * population)

    print(f"Initial objective: {initial_obj}")
    previous_obj = initial_obj

    for ps_iter in range(ps_rounds):
        print(f"\n Profile Steering iteration {ps_iter}")

        for node in nodes:
            node.reset_push_sum()

        # Crash a random subset for this gossip phase
        crashed_nodes = random.sample(nodes, num_crashed) if num_crashed > 0 else []
        for node in crashed_nodes:
            node.crash()

        for gossip_iter in range(gossip_rounds):
            if gossip_iter == crash_duration_rounds:
                for node in crashed_nodes:
                    node.recover()
            for node in nodes:
                node.send()
            for node in nodes:
                node.receive()

        # Ensure full recovery even when crash_duration_rounds >= gossip_rounds
        for node in nodes:
            node.recover()

        # Measure Push-Sum accuracy and track per-node RMSE
        true_x_before_steering = compute_true_aggregate(nodes, profile_length)
        node_rmses = []
        for node in nodes:
            estimate = node.make_estimate()
            rmse = np.sqrt(np.mean([(e - t) ** 2 for e, t in zip(estimate, true_x_before_steering)]))
            node_rmses.append(rmse)
            error = np.linalg.norm([e - t for e, t in zip(estimate, true_x_before_steering)])
            print(f"Node {node.id} estimate error: {error:.4f}")
        rmse_history.append(float(np.mean(node_rmses)))

        # Now do local Profile Steering (skipped for non-steering nodes)
        improvements = []

        for node in nodes:
            if node.id in non_steering_ids:
                continue
            improvement = node.profile_steering_step(alpha)
            improvements.append(improvement)

        true_x = compute_true_aggregate(nodes, profile_length)
        obj = objective(true_x, p)

        objective_history.append(obj)

        print(f"Objective value: {obj}")
        print(f"Total improvement: {sum(improvements)}")

        if abs(previous_obj - obj) < 0.001:
            print("Converged")
            break

        previous_obj = obj

    print("\nFinished!")
    final_x = compute_true_aggregate(nodes, profile_length)
    final_obj = objective(final_x, p)
    aggregate_rmse = float(np.mean(rmse_history))
    print(f"Initial objective: {initial_obj}")
    print(f"Final objective: {final_obj}")
    print(f"Improvement: {initial_obj - final_obj}")
    print(f"Aggregate RMSE (mean over iterations): {aggregate_rmse:.4f}")

    plt.figure()
    plt.plot(range(len(objective_history)), objective_history, marker="o")
    plt.axhline(initial_obj, linestyle="--", label="Initial objective")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Objective value over iterations")
    plt.legend()
    plt.grid(True)
    if plot:
        plt.show()

    return {
        "alpha": alpha,
        "non_steering_proportion": non_steering_proportion,
        "non_steering_count": num_non_steering,
        "crash_fraction": crash_fraction,
        "crash_duration_rounds": crash_duration_rounds,
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement": initial_obj - final_obj,
        "relative_improvement": (initial_obj - final_obj) / initial_obj,
        "iterations": len(objective_history),
        "aggregate_rmse": aggregate_rmse,
    }

if __name__ == "__main__":
    main(alpha=0.075, non_steering_proportion=0.0, plot=True)