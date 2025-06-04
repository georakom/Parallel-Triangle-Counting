from collections import defaultdict
import time

"""
Fennel-inspired streaming partitioning.
Balances edges and sizes with streaming assignments.
It balances partitions with a tunable cost model,
aiming to minimize a graph cut while balancing partition sizes.
"""

def partition_graph(G, num_workers, alpha=1.5):
    start = time.time()

    # Prepare structures to hold partitioning results
    partitions = [[] for _ in range(num_workers)] # Node lists per partition
    assignments = {}                              # Node → partition ID
    part_sizes = [0] * num_workers                # Track sizes of each partition
    neighbor_counts = [defaultdict(int) for _ in range(num_workers)] # Neighbor presence counts per partition

    for u in G.nodes():
        scores = []

        for pid in range(num_workers):
            # Prefer partitions where many of u's neighbors are already assigned
            edge_score = sum(neighbor_counts[pid].get(v, 0) for v in G[u])

            # Penalize large partitions to keep sizes balanced
            size_score = alpha * (part_sizes[pid] ** 1.5)

            # Final score combines locality and load-balance
            scores.append((edge_score - size_score, pid))

        # Pick the partition with the best score
        _, best_pid = max(scores)

        # Assign the node
        assignments[u] = best_pid
        partitions[best_pid].append(u)
        part_sizes[best_pid] += 1

        # Update neighbor counts for this partition
        for v in G[u]:
            neighbor_counts[best_pid][v] += 1

    print(f"Fennel partitioning complete. Assigned {len(G)} nodes to {num_workers} partitions.")
    print(f"Fennel took: {time.time() - start:.4f} seconds.")
    return partitions, assignments
