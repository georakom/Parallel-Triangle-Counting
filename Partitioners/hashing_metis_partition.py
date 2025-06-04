import time
import metis
import networkx as nx
from collections import defaultdict, Counter

"""
Partition the graph using a hybrid strategy:
  - High-degree nodes and their neighbors go to METIS
  - Low-degree nodes follow neighbors' partitions (majority vote), or default to hash
"""

def improved_hybrid_metis_partition(G, num_workers, degree_cutoff=20000):

    start_time = time.time()

    high_deg = set(n for n in G.nodes if G.degree(n) >= degree_cutoff)
    metis_nodes = set(high_deg)
    for u in high_deg:
        metis_nodes.update(G.neighbors(u))

    metis_nodes = list(metis_nodes)
    rest_nodes = list(set(G.nodes()) - set(metis_nodes))

    print(f"METIS will process {len(metis_nodes)} nodes (high-degree and neighbors)")
    print(f"Remaining nodes for neighbor-aware hashing: {len(rest_nodes)}")

    assignments = {}
    partitions = [[] for _ in range(num_workers)]

    if metis_nodes:
        subgraph = G.subgraph(metis_nodes)
        _, parts = metis.part_graph(subgraph, nparts=num_workers)
        for node, part in zip(subgraph.nodes(), parts):
            assignments[node] = part
            partitions[part].append(node)

    # Assign the remaining nodes by neighbor majority or fallback to hash
    for node in rest_nodes:
        neighbor_parts = [assignments[n] for n in G.neighbors(node) if n in assignments]
        if neighbor_parts:
            chosen = Counter(neighbor_parts).most_common(1)[0][0]
        else:
            chosen = hash(node) % num_workers
        assignments[node] = chosen
        partitions[chosen].append(node)

    print(f"Improved hybrid METIS+neighbor-follow took: {time.time() - start_time:.4f} seconds")
    return partitions, assignments