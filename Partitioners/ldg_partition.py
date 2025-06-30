import time
import random

"""
Linear Deterministic Greedy (LDG) Partitioning.
Assign each node to the partition where it shares the most neighbors (connectivity-based), 
with a bias toward smaller partitions (to balance load).
Each node is streamed in, and assigned to the partition that maximizes:
    score = (# neighbors in partition) - (load_penalty)
        -The first term rewards locality (connectivity).
        -The second term penalizes imbalance.
"""

def partition_graph_ldg(G, num_workers, capacity_factor=1.05, balance_penalty = 1.5):
    start = time.time()

    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    partition_sizes = [0] * num_workers
    max_partition_size = int((len(G) / num_workers) * capacity_factor)

    nodes = list(G.nodes())
    random.shuffle(nodes)

    for u in nodes:
        scores = []

        for pid in range(num_workers):
            if partition_sizes[pid] >= max_partition_size:
                scores.append(float('-inf'))
                continue

            neighbor_score = sum(1 for v in G[u] if assignments.get(v) == pid)
            balance_score = balance_penalty * (partition_sizes[pid] / max_partition_size)
            scores.append(neighbor_score - balance_score)

        best_pid = scores.index(max(scores))
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        partition_sizes[best_pid] += 1

    print(f"Improved LDG took: {time.time() - start:.4f} seconds")
    return partitions, assignments
