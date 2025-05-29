import time
from multiprocessing import Pool
import networkx as nx
from collections import defaultdict
import psutil
import os

def extract_all_worker_data(G, partitions, assignments, num_workers):
    worker_edges = [set() for _ in range(num_workers)]
    worker_nodes_used = [set() for _ in range(num_workers)]

    for u, v in G.edges():
        if u > v:
            u, v = v, u
        worker_id = assignments[u]
        worker_edges[worker_id].add((u, v))
        worker_nodes_used[worker_id].update([u, v])

    all_edges = list(G.edges())
    for worker_id in range(num_workers):
        local_nodes = worker_nodes_used[worker_id]
        for u, v in all_edges:
            if u in local_nodes and v in local_nodes:
                if u > v:
                    u, v = v, u
                worker_edges[worker_id].add((u, v))

    worker_data = []
    for worker_id in range(num_workers):
        masters = set(partitions[worker_id])
        used_nodes = {u for edge in worker_edges[worker_id] for u in edge}
        mirrors = used_nodes - masters
        print(f"[Worker {worker_id}] Mirror nodes: {len(mirrors)}")
        worker_data.append((list(worker_edges[worker_id]), masters))
    return worker_data


def edge_iterator_hashed(edge_list, master_nodes):
    process = psutil.Process()
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)
    count = 0
    neighbor_sets = defaultdict(set)
    for u, v in edge_list:
        neighbor_sets[u].add(v)

    for u in master_nodes:
        for v in neighbor_sets[u]:
            count += len(neighbor_sets[u] & neighbor_sets.get(v, set()))

    print(f"[Worker {os.getpid()}] FINISHED. Found {count} triangles. Memory used: {mem_usage_mb:.2f} MB")
    return count


def parallel_triangle_count(G, num_workers, partition_func):
    start = time.time()
    partitions, assignments = partition_func(G, num_workers)
    prep_start = time.time()

    worker_data = extract_all_worker_data(G, partitions, assignments, num_workers)
    print(f"Data preparation took: {time.time() - prep_start:.4f} seconds")
    print(f"Preprocessing took: {time.time() - start:.4f} seconds")

    triangle_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.starmap(edge_iterator_hashed, worker_data)
    print(f"Pure triangle counting took: {time.time() - triangle_time:.4f} seconds")
    return sum(results)

def read_graph_from_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G

if __name__ == "__main__":
    import sys
    from Partitioners import metis_partition # Importing the partitioning algorithm to use

    filepath = "./data/"
    filename = "facebook.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        print(f"Average degree of the graph: {avg_degree:.2f}")
        print(f"Number of Nodes: {graph.number_of_nodes()}")
        print(f"Number of Edges: {graph.number_of_edges()}")

        start_time = time.time()
        total_triangles = parallel_triangle_count(graph, 4, metis_partition) # Deciding the partition
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
        print("Graph file not found.")