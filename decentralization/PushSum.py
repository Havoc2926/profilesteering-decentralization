import random


class Node:

    def __init__(self, id, devices, population, desired_profile=[]):
        self.id = id
        self.devices = devices
        self.desired_profile = desired_profile
        self.population = population

        self.local_profile = [0.0] * len(desired_profile)

        self.mass = [0.0] * len(desired_profile)
        self.weight = 1.0
        self.queue = []
        self.neighbours = []
        self.estimate = []

    def init_profile(self):
        self.local_profile = [0.0] * len(self.desired_profile)

        for device in self.devices:
            profile = device.init(self.desired_profile)
            self.local_profile = [x + y for x, y in zip(profile, self.local_profile)]


    def reset_push_sum(self):
        self.mass = self.local_profile.copy()
        self.weight = 1.0
        self.queue = []
        self.estimate = []

    def add_neighbours(self, neighbours):
        self.neighbours = neighbours

    def send(self):

        mass_send = [x / 2 for x in self.mass]
        weight_send = self.weight / 2

        # keep half locally
        self.mass = [x / 2 for x in self.mass]
        self.weight = self.weight / 2

        # send half to neighbour
        target = random.choice(self.neighbours)
        target.queue.append((mass_send, weight_send))

    def receive(self):
        for mass, weight in self.queue:
            self.mass = list(map(lambda x, y: x + y, self.mass, mass))
            self.weight += weight
        self.queue = []

    def make_estimate(self):
        self.estimate = list(map(lambda x: (x / self.weight) * self.population, self.mass))
        return self.estimate

    def profile_steering_step(self):
        estimated_x = self.make_estimate()
        d = [(x - p) for x, p in zip(estimated_x, self.desired_profile)]

        best_device = None
        best_improvement = 0

        for device in self.devices:
            improvement = device.plan(d)

            if improvement > best_improvement:
                best_improvement = improvement
                best_device = device

        if best_device is not None:
            diff = best_device.accept()
            self.local_profile = [x + y for x, y in zip(self.local_profile, diff)]

        return best_improvement
