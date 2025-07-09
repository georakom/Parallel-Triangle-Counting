import random
import time
import numpy as np

"""
Random Partitioning with Load Balancing.
Assigns nodes randomly to partitions while enforcing capacity constraints.
Provides perfect randomness with controlled imbalance (unlike naive hashing).
"""

def partition_graph_random(adj, num_workers, imbalance_factor=1.1):
    start = time.time()
    n = adj.shape[0]
    avg_nodes = n / num_workers
    max_nodes = int(avg_nodes * imbalance_factor)

    partitions = [[] for _ in range(num_workers)]
    assignments = np.empty(n, dtype=np.int32)
    worker_loads = [0] * num_workers

    for node in range(n):
        valid_workers = [i for i in range(num_workers) if worker_loads[i] < max_nodes]
        if not valid_workers:
            valid_workers = list(range(num_workers))
        chosen = random.choice(valid_workers)
        partitions[chosen].append(node)
        assignments[node] = chosen
        worker_loads[chosen] += 1

    # Ordering same as other partitioners
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(n, dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank
    order_to_node = order

    print(f"Random balanced partitioning took: {time.time() - start:.2f} seconds")
    return partitions, assignments, node_to_order, order_to_node