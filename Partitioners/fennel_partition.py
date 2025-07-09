from collections import defaultdict
import time
import numpy as np

"""
Fennel-inspired streaming partitioning.
Balances edges and sizes with streaming assignments.
It balances partitions with a tunable cost model,
aiming to minimize a graph cut while balancing partition sizes.
"""

def partition_graph(adj, num_workers, alpha=1.5):
    start = time.time()
    N = adj.shape[0]
    partitions = [[] for _ in range(num_workers)]
    assignments = np.empty(N, dtype=np.int32)
    part_sizes = np.zeros(num_workers, dtype=np.int32)
    neighbor_counts = [defaultdict(int) for _ in range(num_workers)]

    for u in range(N):
        scores = []
        u_neighbors = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
        for pid in range(num_workers):
            # Edge score: # of u's neighbors already assigned to this partition
            edge_score = sum(neighbor_counts[pid].get(v, 0) for v in u_neighbors)
            # Size penalty: penalize large partitions to keep them balanced
            size_score = alpha * (part_sizes[pid] ** 1.5)
            scores.append(edge_score - size_score)
        best_pid = np.argmax(scores)
        partitions[best_pid].append(u)
        assignments[u] = best_pid
        part_sizes[best_pid] += 1
        for v in u_neighbors:
            neighbor_counts[best_pid][v] += 1

    # Degree-based node ordering (ascending degree, then id)
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(N), degrees))
    node_to_order = np.empty(N, dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank
    order_to_node = order

    print(f"Fennel-CSR partitioning complete. Assigned {N} nodes to {num_workers} partitions.")
    print(f"Fennel-CSR took: {time.time() - start:.4f} seconds.")
    print("Partition sizes:", [len(p) for p in partitions])
    return partitions, assignments, node_to_order, order_to_node
