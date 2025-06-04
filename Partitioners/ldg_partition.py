import time
from collections import defaultdict

"""
Linear Deterministic Greedy (LDG) Partitioning.
Assign each node to the partition where it shares the most neighbors (connectivity-based), 
with a bias toward smaller partitions (to balance load).
Each node is streamed in, and assigned to the partition that maximizes:
    score = (# neighbors in partition) - (load_penalty)
        -The first term rewards locality (connectivity).
        -The second term penalizes imbalance.
"""

def partition_graph_ldg(G, num_workers, capacity_factor=1.05):
    start = time.time()

    # Initialize partition tracking
    partitions = [[] for _ in range(num_workers)]  # Final partitions
    assignments = {}                               # Node → partition map
    neighbor_counts = [defaultdict(int) for _ in range(num_workers)] # Track neighbors per partition
    partition_sizes = [0] * num_workers            # Track how full each partition is

    # Max partition capacity (slightly over provisioned to allow slack)
    max_partition_size = int((len(G) / num_workers) * capacity_factor)


    for u in G.nodes():
        scores = []

        for pid in range(num_workers):
            if partition_sizes[pid] >= max_partition_size:
                # Partition full — skip it
                scores.append(float("-inf"))
                continue

            # Score: favor neighbor locality, penalize imbalance
            score = neighbor_counts[pid][u] - (partition_sizes[pid] / max_partition_size)
            scores.append(score)

        # Assign node to the best-scoring partition
        best_pid = scores.index(max(scores))
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        partition_sizes[best_pid] += 1

        # Update neighbor counts
        for nbr in G[u]:
            neighbor_counts[best_pid][nbr] += 1

    print(f"LDG partitioning complete. Assigned {len(G)} nodes to {num_workers} partitions.")
    print(f"LDG took: {time.time() - start:.4f} seconds")
    return partitions, assignments
