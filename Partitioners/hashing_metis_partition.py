import time
import metis

"""
Partition the graph using a hybrid strategy:
  - High-degree nodes and their neighbors go to METIS
  - Low-degree nodes are hashed into a partition 
This approach balances partition quality with speed and is much faster than full-METIS while retaining structure.
"""

def improved_hash_metis_partition(G, num_workers, degree_cutoff=40):
    start_time = time.time()

    # Identify high-degree nodes based on cutoff
    high_deg = set(n for n in G.nodes if G.degree(n) >= degree_cutoff)

    # Include neighbors of high-degree nodes in the METIS subgraph
    metis_nodes = set(high_deg)
    for u in high_deg:
        metis_nodes.update(G.neighbors(u))

    metis_nodes = list(metis_nodes)
    rest_nodes = list(set(G.nodes()) - set(metis_nodes)) # Nodes not included in METIS

    print(f"METIS will process {len(metis_nodes)} nodes (high-degree and neighbors)")
    print(f"Remaining nodes for hashing: {len(rest_nodes)}")

    # Assignment dictionary (node → partition id)
    assignments = {}
    partitions = [[] for _ in range(num_workers)]

    # Apply METIS on high-degree subgraph
    if metis_nodes:
        subgraph = G.subgraph(metis_nodes) # Create subgraph for METIS
        _, parts = metis.part_graph(subgraph, nparts=num_workers)
        for node, part in zip(subgraph.nodes(), parts):
            assignments[node] = part
            partitions[part].append(node)

    # Assign the remaining nodes by hashing
    for node in rest_nodes:
        neighbor_ids = sorted([n for n in G.neighbors(node) if n in assignments])
        key = tuple(neighbor_ids[:3])  # use first 3 assigned neighbors for stability
        part = hash(key) % num_workers
        partitions[part].append(node)
        assignments[node] = part

    print(f"Improved hashing-METIS took: {time.time() - start_time:.4f} seconds")
    return partitions, assignments