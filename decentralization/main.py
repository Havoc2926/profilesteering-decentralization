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

def main(alpha=1.0, plot=True):
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

    print(f"Initial objective: {initial_obj}")
    previous_obj = initial_obj

    for ps_iter in range(ps_rounds):
        print(f"\n Profile Steering iteration {ps_iter}")

        for node in nodes:
            node.reset_push_sum()

        for gossip_iter in range(gossip_rounds):
            for node in nodes:
                node.send()

            for node in nodes:
                node.receive()

        # Check Push-Sum accuracy BEFORE steering
        true_x_before_steering = compute_true_aggregate(nodes, profile_length)

        for node in nodes:
            estimate = node.make_estimate()
            error = objective(estimate, true_x_before_steering)
            print(f"Node {node.id} estimate error: {error}")

        # Now do local Profile Steering
        improvements = []

        for node in nodes:

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
    print(f"Initial objective: {initial_obj}")
    print(f"Final objective: {final_obj}")
    print(f"Improvement: {initial_obj - final_obj}")

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
        "initial_objective": initial_obj,
        "final_objective": final_obj,
        "improvement": initial_obj - final_obj,
        "relative_improvement": (initial_obj - final_obj) / initial_obj,
        "iterations": len(objective_history),
    }

if __name__ == "__main__":
    main(alpha=0.075, plot=True)