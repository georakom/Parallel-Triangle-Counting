import time
from collections import defaultdict

"""
Degree-Based Partitioning.
Nodes are sorted by degree and assigned round-robin to partitions.
Better than hashing for locality since high-degree nodes are spread evenly.
Still has significant edge-cut but better than pure random.
"""


def partition_graph_degree(G, num_workers):
    start = time.time()
    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    # Sort nodes by degree (descending)
    nodes_sorted = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)

    for i, node in enumerate(nodes_sorted):
        part = i % num_workers  # Round-robin assignment
        partitions[part].append(node)
        assignments[node] = part

    print(f"Degree partitioning took: {time.time() - start:.4f} seconds")
    return partitions, assignments