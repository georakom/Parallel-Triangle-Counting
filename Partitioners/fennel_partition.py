import random
from collections import defaultdict
import math

"""
Fennel-inspired streaming partitioning.
Balances edges and sizes with streaming assignments.
It balances partitions with a tunable cost model,
aiming to minimize a graph cut while balancing partition sizes.
"""

def partition_graph(G, num_workers, alpha=1.5):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    part_sizes = [0] * num_workers
    neighbor_counts = [defaultdict(int) for _ in range(num_workers)]

    for u in nodes:
        scores = []

        for pid in range(num_workers):
            edge_score = sum(neighbor_counts[pid].get(v, 0) for v in G[u])
            size_score = alpha * (part_sizes[pid] ** 1.5)
            scores.append((edge_score - size_score, pid))

        _, best_pid = max(scores)
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        part_sizes[best_pid] += 1

        for v in G[u]:
            neighbor_counts[best_pid][v] += 1

    print(f"Fennel partitioning complete. Assigned {len(G)} nodes to {num_workers} partitions.")
    return partitions, assignments
