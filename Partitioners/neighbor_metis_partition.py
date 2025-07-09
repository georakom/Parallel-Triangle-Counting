import time
import metis
import numpy as np
import networkx as nx
from collections import Counter

"""
Seeded METIS + Streaming Partitioning.
- Run METIS on a core subgraph (top-K high-degree nodes + their neighbors).
- Stream the rest of the nodes, assigning each to the partition most of its assigned neighbors belong to (or by hash fallback).
"""

def improved_neighbor_metis_partition(adj, num_workers, seed_size=1):
    """
    Partition CSR graph using seeded METIS and streaming assignment.
    - seed_size: Number of high-degree nodes to use as METIS seeds.
    Returns: partitions, assignments, node_to_order, order_to_node
    """
    start = time.time()

    # Convert CSR to NetworkX graph for METIS compatibility
    coo = adj.tocoo()
    G = nx.Graph()
    G.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))
    N = G.number_of_nodes()

    # 1. Choose seed set S: top-K high-degree nodes + their neighbors
    degrees = np.array(adj.sum(axis=1)).flatten()
    # Get top-K nodes by degree
    high_deg_nodes = np.argsort(-degrees)[:seed_size]
    seed_set = set(high_deg_nodes)
    for u in high_deg_nodes:
        seed_set.update(G.neighbors(u))
    # Remove duplicates, sort for reproducibility
    seed_nodes = sorted(seed_set)

    assigned = np.full(N, -1, dtype=np.int32)  # -1 means unassigned
    partitions = [[] for _ in range(num_workers)]

    # 2. Run METIS on the induced subgraph S
    if len(seed_nodes) > 0:
        subgraph = G.subgraph(seed_nodes)
        _, seed_parts = metis.part_graph(subgraph, nparts=num_workers)
        node_to_part = dict(zip(subgraph.nodes(), seed_parts))
        for node, part in node_to_part.items():
            assigned[node] = part
            partitions[part].append(node)
    else:
        node_to_part = {}

    # 3. For each unassigned node, assign to majority neighbor partition or hash fallback
    rest_nodes = [u for u in range(N) if assigned[u] == -1]
    for u in rest_nodes:
        neighbor_parts = [assigned[v] for v in G.neighbors(u) if assigned[v] != -1]
        if neighbor_parts:
            chosen = Counter(neighbor_parts).most_common(1)[0][0]
        else:
            chosen = hash(u) % num_workers
        assigned[u] = chosen
        partitions[chosen].append(u)

    # Generate degree-based node ordering (ascending degree, then node id)
    order = np.lexsort((np.arange(N), degrees))
    node_to_order = np.empty(N, dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank


    print(f"Seeded METIS core: {len(seed_nodes)} nodes; Streamed: {len(rest_nodes)} nodes")
    print(f"Partitioning took {time.time() - start:.2f} seconds")
    print("Partition sizes:", [len(p) for p in partitions])
    return partitions, assigned, node_to_order