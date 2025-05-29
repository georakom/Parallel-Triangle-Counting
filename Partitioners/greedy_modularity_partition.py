import networkx as nx
import time

"""
Distributed Parallel Triangle Counting using Greedy Modularity Communities.
This approach partitions the graph using Clauset-Newman-Moore’s greedy modularity maximization. 
It prioritizes forming dense internal communities quickly, 
which often results in lower mirror counts and strong intra-community triangle locality.
Despite being faster than Louvain, it still incurs significant preprocessing time,
and tends to form a small number of large communities, which may cause load imbalance and hinder scalability.
"""

def partition_graph(G, num_workers):
    """Partition using Greedy Modularity and round-robin assignment."""
    start = time.time()
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    print(f"Greedy Modularity took: {time.time() - start:.4f} seconds")

    partitions = [[] for _ in range(num_workers)]
    assignments = {}

    for i, comm in enumerate(communities):
        worker_id = i % num_workers
        for node in comm:
            partitions[worker_id].append(node)
            assignments[node] = worker_id

    print(f"Found {len(communities)} communities, assigned to {num_workers} workers")
    return partitions, assignments
