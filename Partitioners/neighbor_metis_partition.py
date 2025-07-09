import time
import metis
from collections import  Counter
import numpy as np
import networkx as nx

"""
Partition the graph using a hybrid strategy:
  - High-degree nodes and their neighbors go to METIS
  - Low-degree nodes follow neighbors' partitions (majority vote)
This approach balances partition quality with speed and is much faster than full-METIS while retaining structure.
"""

def improved_neighbor_metis_partition(adj, num_workers, degree_cutoff=10000):
    """
        Partition CSR graph using hybrid strategy:
        - METIS on high-degree nodes + their neighbors
        - Remaining nodes assigned via neighbor voting or hash fallback
        Returns: partitions, assignments, node_to_order, order_to_node
        """
    start = time.time()

    # Convert CSR to NetworkX graph
    coo = adj.tocoo()
    G = nx.Graph()
    G.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))
    G.remove_edges_from(nx.selfloop_edges(G))

    # Identify high-degree nodes and their neighbors
    high_deg = {n for n, d in G.degree if d >= degree_cutoff}
    metis_nodes = set(high_deg)
    for u in high_deg:
        metis_nodes.update(G.neighbors(u))

    rest_nodes = set(G.nodes()) - metis_nodes
    print(f"METIS will process {len(metis_nodes)} nodes (high-degree + neighbors)")
    print(f"Remaining nodes for neighbor-aware hashing: {len(rest_nodes)}")

    # Partition containers
    assignments = np.empty(adj.shape[0], dtype=np.int32)
    partitions = [[] for _ in range(num_workers)]

    # Run METIS on subgraph
    if metis_nodes:
        subgraph = G.subgraph(metis_nodes)
        _, parts = metis.part_graph(subgraph, nparts=num_workers)
        for node, part in zip(subgraph.nodes(), parts):
            assignments[node] = part
            partitions[part].append(node)

    # Assign remaining nodes based on neighbor vote or hash fallback
    for node in rest_nodes:
        neighbor_parts = [assignments[n] for n in G.neighbors(node) if n in metis_nodes]
        if neighbor_parts:
            chosen = Counter(neighbor_parts).most_common(1)[0][0]
        else:
            chosen = hash(node) % num_workers
        assignments[node] = chosen
        partitions[chosen].append(node)

    # Generate node ordering (same logic as other partitioners)
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(adj.shape[0], dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank
    order_to_node = order

    print(f"Hybrid METIS+neighbor partitioning took: {time.time() - start:.2f} seconds")
    return partitions, assignments, node_to_order, order_to_node