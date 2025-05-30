import time
import metis

""" 
Distributed Parallel Triangle Counting using METIS. 
While pure triangle-counting is extremely fast, we have a very noticeable preprocessing and memory overhead. 
Due to METIS non triangle-aware partitioning, we end up losing scalability, especially in memory.
"""

def partition_graph(G, num_workers):
    start = time.time()
    _, parts = metis.part_graph(G, nparts=num_workers)
    print(f"METIS took: {time.time() - start:.4f} seconds")
    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    for node, part in zip(G.nodes(), parts):
        partitions[part].append(node)
        assignments[node] = part
    return partitions, assignments

