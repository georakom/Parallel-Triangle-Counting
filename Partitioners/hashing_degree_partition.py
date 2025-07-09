import time
import numpy as np
"""
Hybrid Hashing + Degree Chunking.
Groups nodes into degree-sorted chunks, then hashes chunks to partitions.
Attempts to balance the benefits of degree-aware and hashed partitioning (triangle balance and speed)
"""

def partition_graph_hybrid(adj, num_workers):
    start = time.time()
    N = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(N), degrees))
    node_to_order = np.empty(N, dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank

    partitions = [[] for _ in range(num_workers)]
    assignments = np.empty(N, dtype=np.int32)

    # Hash assignment using degree order
    for node in order:
        wid = hash(node) % num_workers
        partitions[wid].append(node)
        assignments[node] = wid

    print(f"Hashing by degree partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments, node_to_order