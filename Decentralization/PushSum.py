
class Node:

    def __init__(self, id, profile, population):
        self.id = id
        self.profile = profile
        self.weight = 1
        self.population = population
        self.queue = []
        self.neighbours = []
        self.estimate = []

    def add_neighbours(self, neighbours):
        self.neighbours = neighbours

    def send(self):

        divisor = self.population + 1
        profile_send = list(map(lambda x: x / divisor, self.profile))
        weight_send = self.weight / divisor

        self.queue.append((profile_send, weight_send))

        for node in self.neighbours:
            node.queue.append((profile_send, weight_send))

    def receive(self):
        for profile, weight in self.queue:
            self.profile = list(map(lambda x, y: x + y, self.profile, profile))
            self.weight += weight
        self.queue = []

    def makeEstimate(self):
        self.estimate = list(map(lambda x: (x / self.weight) * self.population, self.profile))
        return self.estimate

def run():
    # Create the nodes
    node1 = Node(1, [100, 200, 300], 5)
    node2 = Node(2, [400, 500, 600], 5)
    node3 = Node(3, [700, 800, 900], 5)
    node4 = Node(4, [1000, 1100, 1200], 5)
    node5 = Node(5, [1300, 1400, 1500], 5)

    # Connect the nodes
    node1.add_neighbours([node2, node3, node5])
    node2.add_neighbours([node1, node3, node5])
    node3.add_neighbours([node1, node2, node4, node5])
    node4.add_neighbours([node3, node5])
    node5.add_neighbours([node1, node2, node3, node4])

    nodes = [node1, node2, node3, node4, node5]

    print("True aggregate is [3500, 4000, 4500]")

    # Run the algorithm for a number of iterations
    for i in range(200):
        print("Iteration", i)
        for node in nodes:
            node.send()
        for node in nodes:
            node.receive()
        for j, node in enumerate(nodes):
            print(f"Node {j+1} estimate: {node.makeEstimate()}")

    print("===============================================================================================================")

    print("PUSHSUM DONE!")

    for node in nodes:
        print(f"Discrepancy in node {node.id}'s estimate: {list(map(lambda x, y: x - y, node.makeEstimate(), [3500, 4000, 4500]))}")

run()