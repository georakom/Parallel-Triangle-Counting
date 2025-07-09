import networkx as nx
import time
import numpy as np

"""
This method uses asynchronous Label Propagation to detect communities, which are then assigned to workers
in a round-robin fashion.  
Label Propagation tends to create many small, unstable communities that lead to a high number of mirror nodes 
and a poor balance between workers.
This can hurt both memory usage and triangle-counting efficiency in distributed settings.
"""

def partition_graph(adj, num_workers):

    start = time.time()

    # Convert CSR to NetworkX
    coo = adj.tocoo()
    G = nx.Graph()
    G.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))

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

    # Build node ordering: here we use degree-ordered, but you could use another ordering if you want
    degrees = dict(G.degree())
    order = sorted(G.nodes(), key=lambda n: (degrees[n], n))
    node_to_order = {}
    for rank, node in enumerate(order):
        node_to_order[node] = rank

    print(f"Found {len(communities)} communities, assigned to {num_workers} workers")
    print(f"Label Propagation took: {time.time() - start:.4f} seconds")
    return partitions, assignments, node_to_order