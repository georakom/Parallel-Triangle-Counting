import time
import random
from collections import defaultdict

def partition_graph_ldg(G, num_workers, capacity_factor=1.05):
    """
    Linear Deterministic Greedy (LDG) Partitioning.
    Each node is streamed in, and assigned to the partition that maximizes:
        score = (# neighbors in partition) - (load_penalty)
    """
    start = time.time()
    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    neighbor_counts = [defaultdict(int) for _ in range(num_workers)]
    partition_sizes = [0] * num_workers
    max_partition_size = int((len(G) / num_workers) * capacity_factor)

    nodes = list(G.nodes())
    random.shuffle(nodes)

    for u in nodes:
        scores = []
        for pid in range(num_workers):
            if partition_sizes[pid] >= max_partition_size:
                scores.append(float("-inf"))
                continue
            score = neighbor_counts[pid][u] - (partition_sizes[pid] / max_partition_size)
            scores.append(score)

        best_pid = scores.index(max(scores))
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        partition_sizes[best_pid] += 1

        for nbr in G[u]:
            neighbor_counts[best_pid][nbr] += 1

    print(f"LDG partitioning complete. Assigned {len(G)} nodes to {num_workers} partitions in {time.time() - start:.4f} seconds.")
    return partitions, assignments
