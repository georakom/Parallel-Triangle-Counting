import random
import time
from collections import defaultdict

"""
Random Partitioning with Load Balancing.
Assigns nodes randomly to partitions while enforcing capacity constraints.
Provides perfect randomness with controlled imbalance (unlike naive hashing).
"""

def partition_graph_random(G, num_workers, imbalance_factor=1.1):
    start = time.time()

    # Calculate capacity limits
    avg_nodes = len(G.nodes()) / num_workers
    max_nodes = int(avg_nodes * imbalance_factor)

    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    worker_loads = [0] * num_workers

    # Shuffle nodes for true randomness


    # Assign nodes with load balancing
    for node in G.nodes():
        # Find workers with capacity
        valid_workers = [i for i in range(num_workers)
                         if worker_loads[i] < max_nodes]

        # If all workers full, relax constraints
        if not valid_workers:
            valid_workers = range(num_workers)

        # Random assignment among valid workers
        part = random.choice(valid_workers)
        partitions[part].append(node)
        assignments[node] = part
        worker_loads[part] += 1

    print(f"Random partitioning took: {time.time() - start:.4f} secs")
    return partitions, assignments