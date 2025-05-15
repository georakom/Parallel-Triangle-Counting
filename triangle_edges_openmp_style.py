import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict

def build_A_plus(G, node_ranking):
    A_plus = defaultdict(list)
    for u, v in G.edges():
        if node_ranking[u] < node_ranking[v]:
            A_plus[u].append(v)
        else:
            A_plus[v].append(u)
    for node in A_plus:
        A_plus[node].sort()
    return A_plus

def tc_merge_worker(edges, A_plus):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(edges)} edges")

    triangle_count = 0
    for u, v in edges:
        neighbors_u = A_plus[u]
        neighbors_v = A_plus[v]
        i, j = 0, 0
        while i < len(neighbors_u) and j < len(neighbors_v):
            if neighbors_u[i] == neighbors_v[j]:
                triangle_count += 1
                i += 1
                j += 1
            elif neighbors_u[i] < neighbors_v[j]:
                i += 1
            else:
                j += 1

    print(f"[Worker {pid}] FINISHED. Found {triangle_count} triangles.", flush=True)
    return triangle_count

# Wrapper for imap_unordered
def tc_merge_wrapper(args):
    return tc_merge_worker(*args)

def parallel_triangle_count(G, num_workers):
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    node_ranking = {node: i for i, node in enumerate(nodes_sorted_by_degree)}
    A_plus = build_A_plus(G, node_ranking)

    edge_list = [(u, v) for u in A_plus for v in A_plus[u]]

    # Dynamic scheduling with small chunks
    chunk_size = 5000
    chunks = [edge_list[i:i + chunk_size] for i in range(0, len(edge_list), chunk_size)]
    task_args = [(chunk, A_plus) for chunk in chunks]

    with Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(tc_merge_wrapper, task_args)
        total_triangles = sum(results)

    return total_triangles

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
    filename = "Email-Enron.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

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