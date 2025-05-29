"""
Distributed Parallel Triangle Counting using Infomap Community Detection.

Infomap uses random walks to identify communities where information flows efficiently. It often produces natural,
high-modularity partitions that are beneficial for reducing mirror nodes and maintaining triangle locality.

While more stable than Label Propagation and often faster than Louvain, Infomap still incurs moderate preprocessing cost
and may generate many small communities, requiring additional logic for load balancing in distributed processing.
"""

from infomap import Infomap
from collections import defaultdict

def partition_graph(G, num_workers):
    """
    Partition the graph using Infomap and assign communities to workers in round-robin fashion.
    Returns: (partitions, assignments)
    """
    import time
    start = time.time()

    im = Infomap("--two-level")

    for u, v in G.edges():
        im.addLink(u, v)

    im.run()

    # Group nodes by community ID
    communities = defaultdict(list)
    for node in im.nodes:
        communities[node.moduleIndex()].append(node.node_id)

    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    for i, comm in enumerate(communities.values()):
        worker_id = i % num_workers
        partitions[worker_id].extend(comm)
        for node in comm:
            assignments[node] = worker_id

    print(f"Infomap partitioning took: {time.time() - start:.4f} seconds")
    print(f"Found {len(communities)} communities, assigned to {num_workers} workers")
    return partitions, assignments
