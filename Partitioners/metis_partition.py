import time
import metis

""" 
While pure triangle-counting is extremely fast, we have a very noticeable preprocessing and memory overhead. 
Due to METIS non triangle-aware partitioning, we end up losing scalability, especially in memory.
"""

def partition_graph(G, num_workers):
    start = time.time()

    # Use METIS to partition the graph into number of workers parts
    _, parts = metis.part_graph(G, nparts=num_workers)

    # Organize nodes into partitions
    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    for node, part in zip(G.nodes(), parts):
        partitions[part].append(node)
        assignments[node] = part
    print(f"METIS took: {time.time() - start:.4f} seconds")
    return partitions, assignments

