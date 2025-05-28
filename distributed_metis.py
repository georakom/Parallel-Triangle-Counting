import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict
import metis
import psutil

""" 
Distributed Parallel Triangle Counting using METIS. 
While pure triangle-counting is extremely fast, we have a very noticeable preprocessing and memory overhead. 
Due to METIS non triangle-aware partitioning, we end up losing scalability, especially in memory 
(the more cores/workers we assign, the more total mirrors we end up having).
"""
def partition_graph(G, num_workers):
    """Use METIS to partition an undirected graph. Returns partitions and assignments."""
    metis_time = time.time()
    _, parts = metis.part_graph(G, nparts=num_workers)
    print(f"METIS took: {time.time() - metis_time:.4f} seconds")
    partitions = [[] for _ in range(num_workers)]
    assignments = {}
    for node, part in zip(G.nodes(), parts):
        partitions[part].append(node)
        assignments[node] = part
    return partitions, assignments

# def count_mirror_nodes(G, partitions, assignments):
#     """
#     For each worker, count mirror nodes based on master nodes' neighbors
#     """
#     total_mirrors = 0
#     for worker_id, master_nodes in enumerate(partitions):
#         master_set = set(master_nodes)
#         mirror_set = set()
#
#         for u in master_nodes:
#             for v in G[u]:  # neighbors
#                 if v not in master_set:
#                     mirror_set.add(v)
#
#         print(f"[Worker {worker_id}] Mirror nodes: {len(mirror_set)}")
#         total_mirrors += len(mirror_set)
#
#     print(f"\nTotal mirror nodes: {total_mirrors}")
#     return total_mirrors

def extract_all_worker_data(G, partitions, assignments, num_workers):
    """
    Optimized version: builds all edge lists in a single pass.
    """
    # Prepare per-worker storage
    worker_edges = [set() for _ in range(num_workers)]
    worker_nodes_used = [set() for _ in range(num_workers)]

    for u, v in G.edges():
        # Simulate DAG direction: always go u → v if u < v
        if u > v:
            u, v = v, u

        worker_id = assignments[u]
        worker_edges[worker_id].add((u, v))
        worker_nodes_used[worker_id].update([u, v])

    # Proxy edges: any edge between nodes used by a worker should be included
    all_edges = list(G.edges())
    for worker_id in range(num_workers):
        local_nodes = worker_nodes_used[worker_id]
        for u, v in all_edges:
            if u in local_nodes and v in local_nodes:
                if u > v:
                    u, v = v, u
                worker_edges[worker_id].add((u, v))

    # Track master/mirror sets
    worker_data = []
    for worker_id in range(num_workers):
        masters = set(partitions[worker_id])
        used_nodes = {u for edge in worker_edges[worker_id] for u in edge}
        mirrors = used_nodes - masters
        print(f"[Worker {worker_id}] Mirror nodes: {len(mirrors)}")
        worker_data.append((list(worker_edges[worker_id]), masters))

    return worker_data


def edge_iterator_hashed(edge_list, master_nodes):
    process = psutil.Process(os.getpid())
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)
    triangle_count = 0

    neighbor_sets = defaultdict(set)
    for u, v in edge_list:
        neighbor_sets[u].add(v)

    for u in master_nodes:
        for v in neighbor_sets[u]:
            triangle_count += len(neighbor_sets[u] & neighbor_sets.get(v, set()))

    print(f"[Worker {os.getpid()}] FINISHED. Found {triangle_count} triangles. Memory used: {mem_usage_mb:.2f} MB", flush=True)
    return triangle_count

def parallel_triangle_count(G, num_workers):
    start_pre_time = time.time()

    # Step 1: Partitioning
    partitions, assignments = partition_graph(G, num_workers)
    #total_mirrors = count_mirror_nodes(G, partitions, assignments)
    prepare_time = time.time()

    # Step 2: Prepare edge sets for each worker
    worker_data = extract_all_worker_data(G, partitions, assignments, num_workers)

    print(f"Data preparation took: {time.time() - prepare_time:.4f} seconds")
    end_pre_time = time.time()
    print(f"Preprocessing took: {end_pre_time - start_pre_time:.6f} seconds")
    triangle_time = time.time()

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(edge_iterator_hashed, worker_data)
    triangle_finish = time.time()

    print(f"Pure triangle counting took: {triangle_finish - triangle_time:.4f} seconds")
    return sum(results)

def read_graph_from_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    random.shuffle(edges)
    for u, v in edges:
        G.add_edge(u, v)
    return G

if __name__ == "__main__":
    filepath = "./data/"
    filename = "amazon.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        print(f"Average degree of the graph: {avg_degree:.2f}")
        print(f"Number of Nodes: {graph.number_of_nodes()}")
        print(f"Number of Edges: {graph.number_of_edges()}")

        start_time = time.time()
        total_triangles = parallel_triangle_count(graph, 4)
        end_time = time.time()

        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)

        start_time = time.time()
        triangles_networkx = nx.triangles(graph)
        triangles_networkx_count = sum(triangles_networkx.values()) // 3
        end_time = time.time()
        print(f"Total triangles (NetworkX): {triangles_networkx_count}")
        print("Triangle Algorithm time (NetworkX): ", end_time - start_time)

    except FileNotFoundError:
        print("File not found")
