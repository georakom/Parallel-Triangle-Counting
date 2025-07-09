import time
import numpy as np
"""
Degree-Based Partitioning.
Nodes are sorted by degree and assigned round-robin to partitions.
Better than hashing for locality since high-degree nodes are spread evenly.
Still has significant edge-cut but better than pure random.
"""

def partition_graph_degree(adj, num_workers):
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(adj.shape[0], dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank

    partitions = [[] for _ in range(num_workers)]
    assignments = np.empty(adj.shape[0], dtype=np.int32)
    for i, node in enumerate(order):
        wid = i % num_workers
        partitions[wid].append(node)
        assignments[node] = wid

    return partitions, assignments, node_to_order