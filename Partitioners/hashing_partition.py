import time
import numpy as np

"""
Distributed Partitioning using Node Hashing.
The fastest possible partitioning method - simply hashes each node to a partition.
Provides efficient load balancing but terrible locality (high edge-cut).
Useful when speed is critical and communication overhead is acceptable.
"""

def partition_graph_hash(adj_csr, num_workers):
    start = time.time()

    N = adj_csr.shape[0]
    partitions = [[] for _ in range(num_workers)]
    assignments = np.zeros(N, dtype=int)

    for i in range(N):
        p = i % num_workers
        partitions[p].append(i)
        assignments[i] = p

    # For consistency, use node id as the ordering (could also use degree)
    node_to_order = np.arange(N, dtype=np.int32)

    print(f"Hashing partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments, node_to_order