import time

"""
Hybrid Hashing + Degree Chunking.
Groups nodes into degree-sorted chunks, then hashes chunks to partitions.
Attempts to balance the benefits of degree-aware and hashed partitioning.
"""

def partition_graph_hybrid(G, num_workers, chunk_size=1000):
    start = time.time()

    # Initialize containers
    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    # Sort nodes by degree and split into chunks
    nodes_sorted = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    chunks = [nodes_sorted[i:i + chunk_size] for i in range(0, len(nodes_sorted), chunk_size)]

    # Hash each chunk to assign it to a partition
    for chunk in chunks:
        part = hash(tuple(chunk)) % num_workers  # Hash entire chunk
        partitions[part].extend(chunk)
        for node in chunk:
            assignments[node] = part

    print(f"Hybrid partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments