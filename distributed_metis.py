import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict
import metis
import psutil

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

def extract_partition_edges(G, master_nodes, assignments, worker_id):
    """
    For a given worker, assign edges based on simulated DAG (u < v), and include triangle-completing edges.
    """
    local_edges = []
    local_nodes = set(master_nodes)

    for u, v in G.edges():
        if u < v:
            src, dst = u, v
        else:
            src, dst = v, u

        if assignments[src] == worker_id:
            local_edges.append((src, dst))
            local_nodes.add(dst)

    # Include proxy edges between local-local nodes
    all_edges = set(local_edges)
    for u, v in G.edges():
        if u in local_nodes and v in local_nodes:
            u_dag, v_dag = (u, v) if u < v else (v, u)
            all_edges.add((u_dag, v_dag))

    return list(all_edges), set(master_nodes)

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

    prepare_time = time.time()
    # Step 2: Prepare edge sets for each worker
    worker_data = []
    for worker_id in range(num_workers):
        master_nodes = partitions[worker_id]
        edge_list, master_set = extract_partition_edges(G, master_nodes, assignments, worker_id)
        worker_data.append((edge_list, master_set))
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
