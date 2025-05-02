import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict
import metis

def metis_partition(G, num_workers):
    """Partitions the undirected graph using METIS before DAG construction."""
    (edgecuts, parts) = metis.part_graph(G, nparts=num_workers)
    node_to_worker = {node: part for node, part in zip(G.nodes(), parts)}
    partitions = [[] for _ in range(num_workers)]
    for node, worker in node_to_worker.items():
        partitions[worker].append(node)
    print(f"METIS partitioning completed with {num_workers} partitions")
    return partitions, node_to_worker

def preprocess_graph(G):
    """Builds DAG and sorted adjacency."""
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    node_ranking = {node: i for i, node in enumerate(nodes_sorted_by_degree)}
    DAG = nx.DiGraph()
    sorted_adjacency = {}

    for u, v in G.edges():
        if node_ranking[u] < node_ranking[v]:
            DAG.add_edge(u, v)
        else:
            DAG.add_edge(v, u)

    for node in DAG.nodes():
        sorted_adjacency[node] = sorted(DAG.successors(node), key=lambda x: node_ranking[x])

    return DAG, node_ranking, sorted_adjacency

def count_mirrors(sorted_adjacency, node_to_worker):
    mirror_count = defaultdict(int)
    total_mirrors = 0
    for node, neighbors in sorted_adjacency.items():
        owner = node_to_worker[node]
        for neighbor in neighbors:
            if node_to_worker[neighbor] != owner:
                mirror_count[neighbor] += 1
                total_mirrors += 1
    return mirror_count, total_mirrors

def compact_forward_count(nodes, sorted_adjacency, node_ranking):
    worker_pid = os.getpid()
    print(f"[Worker {worker_pid}] STARTED with {len(nodes)} nodes")

    triangle_count = 0
    for v in nodes:
        v_neighbors = sorted_adjacency[v]
        for u in v_neighbors:
            if node_ranking[u] > node_ranking[v]:
                u_neighbors = sorted_adjacency[u]
                v_ptr, u_ptr = 0, 0
                while v_ptr < len(v_neighbors) and u_ptr < len(u_neighbors):
                    if v_neighbors[v_ptr] == u_neighbors[u_ptr]:
                        triangle_count += 1
                        v_ptr += 1
                        u_ptr += 1
                    elif node_ranking[v_neighbors[v_ptr]] < node_ranking[u_neighbors[u_ptr]]:
                        v_ptr += 1
                    else:
                        u_ptr += 1

    print(f"[Worker {worker_pid}] FINISHED. Found {triangle_count} triangles.", flush=True)
    return triangle_count

def parallel_triangle_count(G, num_workers):
    start_sorting_time = time.time()
    partitions, node_to_worker = metis_partition(G, num_workers)
    DAG, node_ranking, sorted_adjacency = preprocess_graph(G)

    mirror_count, total_mirrors = count_mirrors(sorted_adjacency, node_to_worker)
    print(f"Total mirrors created: {total_mirrors}")
    print(f"Average mirrors per node: {total_mirrors / len(sorted_adjacency):.2f}")
    print(f"Sorting/Preprocessing/Partitioning time: {time.time() - start_sorting_time:.4f} seconds")

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(compact_forward_count, [(chunk, sorted_adjacency, node_ranking) for chunk in partitions])

    return sum(results), mirror_count, total_mirrors

def read_graph_from_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    random.shuffle(edges)
    for u, v in edges:
        G.add_edge(u, v)
    return G

if __name__ == "__main__":

    filepath = "C:/Users/Jorjm/OneDrive/Desktop/Triangle_Counting/data/"
    filename = "facebook.txt"

    try:
        graph = read_graph_from_file(filepath + filename)
        start_time = time.time()
        total_triangles, mirror_count, total_mirrors = parallel_triangle_count(graph, 4)   
        end_time = time.time()

        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)

        start_time = time.time()
        triangles_networkx = nx.triangles(graph)
        triangles_networkx_count = sum(triangles_networkx.values()) // 3
        end_time = time.time()
        print(f"Total triangles (NetworkX): {triangles_networkx_count}")
        print("Triangle Algorithm time (NetworkX): ", end_time - start_time)
    except:
        print("File not found")