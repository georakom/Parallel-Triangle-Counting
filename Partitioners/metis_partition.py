import time
import numpy as np
import metis
import networkx as nx

def metis_partition(adj, num_workers):
    """
    Partition the graph using METIS and return results in the format:
    (partitions, assignments, node_to_order, order_to_node)
    Compatible with triangle counting code.
    """
    start = time.time()
    # Convert CSR to NetworkX graph
    coo = adj.tocoo()
    G = nx.Graph()
    G.add_edges_from(zip(coo.row.tolist(), coo.col.tolist()))

    # Ensure no self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # Partition using METIS
    _, parts = metis.part_graph(G, nparts=num_workers)

    # Organize output
    partitions = [[] for _ in range(num_workers)]
    assignments = np.empty(adj.shape[0], dtype=np.int32)
    for node, part in zip(G.nodes(), parts):
        partitions[part].append(node)
        assignments[node] = part

    # Use degree-based ordering for node_to_order
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = np.empty(adj.shape[0], dtype=np.int32)
    for rank, node in enumerate(order):
        node_to_order[node] = rank

    print(f"METIS partitioning took: {time.time() - start:.2f} seconds")
    print("Partition sizes:", [len(p) for p in partitions])
    return partitions, assignments, node_to_order