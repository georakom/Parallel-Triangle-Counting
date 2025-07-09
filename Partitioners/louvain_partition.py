import time
from collections import defaultdict
from community import community_louvain
import numpy as np
import networkx as nx
"""
Distributed Parallel Triangle Counting using Louvain Community detection.
The Louvain algorithm hierarchically detects communities by optimizing modularity.
It often produces compact, high-quality clusters, which helps reduce mirror nodes,
and balance the triangle load well across partitions.
However, Louvain is relatively slow due to its global optimization phase. 
For medium to large graphs, the preprocessing time dominates, 
making it less suitable for time-critical distributed systems.
"""

def partition_graph(adj, num_workers):
    """Louvain partitioning with smart assignment of large communities."""
    start = time.time()

    # Convert CSR to NetworkX
    G_nx = nx.from_scipy_sparse_matrix(adj)

    # Run Louvain algorithm
    partition = community_louvain.best_partition(G_nx, resolution=1.0, random_state=42)

    # Group by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    sorted_communities = sorted(communities.values(), key=len, reverse=True)

    partitions = [[] for _ in range(num_workers)]
    assignments = np.zeros(adj.shape[0], dtype=np.int32)

    worker_loads = [0] * num_workers
    for comm_nodes in sorted_communities:
        min_worker = worker_loads.index(min(worker_loads))
        for node in comm_nodes:
            partitions[min_worker].append(node)
            assignments[node] = min_worker
        worker_loads[min_worker] += len(comm_nodes)

    # Create default ordering: degree-based
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(adj.shape[0], dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank

    print(f"Assigned {len(sorted_communities)} communities to {num_workers} workers.")
    print(f"Louvain partitioning took: {time.time() - start:.2f} seconds")

    return partitions, assignments, node_to_order


