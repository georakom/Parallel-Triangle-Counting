import time
from collections import defaultdict
import community as community_louvain

"""
Distributed Parallel Triangle Counting using Louvain Community detection.
The Louvain algorithm hierarchically detects communities by optimizing modularity.
It often produces compact, high-quality clusters, which helps reduce mirror nodes,
and balance the triangle load well across partitions.
However, Louvain is relatively slow due to its global optimization phase. 
For medium to large graphs, the preprocessing time dominates, 
making it less suitable for time-critical distributed systems.
"""

def partition_graph(G, num_workers):
    """Louvain partitioning with smart assignment of large communities."""
    start = time.time()

    partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)

    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    sorted_communities = sorted(communities.values(), key=len, reverse=True)

    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    # Assign largest communities first to least-loaded worker
    worker_loads = [0] * num_workers
    for comm_nodes in sorted_communities:
        min_worker = worker_loads.index(min(worker_loads))
        partitions[min_worker].extend(comm_nodes)
        worker_loads[min_worker] += len(comm_nodes)
        for node in comm_nodes:
            assignments[node] = min_worker


    print(f"Assigned {len(sorted_communities)} communities to {num_workers} workers.")
    print(f"Louvain partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments


