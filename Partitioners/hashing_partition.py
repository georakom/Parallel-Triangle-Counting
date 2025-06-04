import time

"""
Distributed Partitioning using Node Hashing.
The fastest possible partitioning method - simply hashes each node to a partition.
Provides efficient load balancing but terrible locality (high edge-cut).
Useful when speed is critical and communication overhead is acceptable.
"""

def partition_graph_hash(G, num_workers):
    start = time.time()

    # Initialize empty partitions and assignment map
    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    # Assign each node to a partition by hashing its ID
    for node in G.nodes():
        part = hash(node) % num_workers  # Simple hash assignment
        partitions[part].append(node)
        assignments[node] = part

    print(f"Hashing partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments