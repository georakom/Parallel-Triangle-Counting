import networkx as nx
import time

"""
This method uses asynchronous Label Propagation to detect communities, which are then assigned to workers
in a round-robin fashion.  
Label Propagation tends to create many small, unstable communities that lead to a high number of mirror nodes 
and a poor balance between workers.
This can hurt both memory usage and triangle-counting efficiency in distributed settings.
"""

def partition_graph(G, num_workers):
    """Partition using Label Propagation and round-robin assignment."""
    start = time.time()

    # Detect communities using LPA
    communities = list(nx.algorithms.community.label_propagation.asyn_lpa_communities(G, seed=42))

    # Prepare partition containers
    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    # Assign each community to a worker (round-robin for balance)
    for i, comm in enumerate(communities):
        worker_id = i % num_workers
        for node in comm:
            partitions[worker_id].append(node)
            assignments[node] = worker_id

    print(f"Found {len(communities)} communities, assigned to {num_workers} workers")
    print(f"Label Propagation took: {time.time() - start:.4f} seconds")
    return partitions, assignments
