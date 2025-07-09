import time
import random
import numpy as np

"""
Linear Deterministic Greedy (LDG) Partitioning.
Assign each node to the partition where it shares the most neighbors (connectivity-based), 
with a bias toward smaller partitions (to balance load).
Each node is streamed in, and assigned to the partition that maximizes:
    score = (# neighbors in partition) - (load_penalty)
        -The first term rewards locality (connectivity).
        -The second term penalizes imbalance.
"""

def partition_graph_ldg(adj, num_workers, capacity_factor=1.05, balance_penalty = 1.5):
    """
    Linear Deterministic Greedy (LDG) for CSR input.
    Assigns each node to the partition with highest connectivity + load penalty.
    """
    start = time.time()
    n = adj.shape[0]
    max_partition_size = int((n / num_workers) * capacity_factor)

    partitions = [[] for _ in range(num_workers)]
    assignments = np.full(n, -1, dtype=np.int32)
    partition_sizes = [0] * num_workers

    nodes = list(range(n))
    random.shuffle(nodes)

    for u in nodes:
        scores = []
        u_neighbors = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]

        for pid in range(num_workers):
            if partition_sizes[pid] >= max_partition_size:
                scores.append(float('-inf'))
                continue

            neighbor_score = sum(1 for v in u_neighbors if assignments[v] == pid)
            balance_score = balance_penalty * (partition_sizes[pid] / max_partition_size)
            scores.append(neighbor_score - balance_score)

        best_pid = scores.index(max(scores))
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        partition_sizes[best_pid] += 1

    # Degree-based ordering
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(n, dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank
    order_to_node = order

    print(f"LDG partitioning took: {time.time() - start:.2f} seconds")
    return partitions, assignments, node_to_order, order_to_node

