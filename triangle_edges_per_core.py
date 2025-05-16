import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict

# Shared-Memory Parallel Triangle Counting with TC-Merge and TC-Hash
# Work between cores is split based on num_edges / cores

def build_A_plus(G, node_ranking):
    A_plus = defaultdict(list)
    for u, v in G.edges():
        if node_ranking[u] < node_ranking[v]:
            A_plus[u].append(v)
        else:
            A_plus[v].append(u)
    for node in A_plus:
        A_plus[node].sort()  # Needed for merge-based intersection
    return A_plus

def build_A_sets(A_plus):
    return {node: set(neighs) for node, neighs in A_plus.items()}

def tc_merge_worker(edges, A_plus):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED (MERGE) with {len(edges)} edges")

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

    print(f"[Worker {pid}] FINISHED (MERGE). Found {triangle_count} triangles.", flush=True)
    return triangle_count

def tc_hash_worker(edges, A_sets):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED (HASH) with {len(edges)} edges")

    triangle_count = 0
    for u, v in edges:
        neighbors_u = A_sets.get(u)
        neighbors_v = A_sets.get(v)
        if neighbors_u and neighbors_v:
            triangle_count += len(neighbors_u & neighbors_v)

    print(f"[Worker {pid}] FINISHED (HASH). Found {triangle_count} triangles.", flush=True)
    return triangle_count

def parallel_triangle_count(G, num_workers, method="merge"):
    # Build node ranking
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x))
    node_ranking = {node: i for i, node in enumerate(nodes_sorted_by_degree)}
    
    A_plus = build_A_plus(G, node_ranking)
    edge_list = [(u, v) for u in A_plus for v in A_plus[u]]

    # Split edges evenly across cores
    chunk_size = len(edge_list) // num_workers
    partitions = [edge_list[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers - 1)] + [edge_list[(num_workers - 1)*chunk_size:]]

    #Or we can use smaller edge chunks, Open-MP style
    #chunk_size = 5000
    #partitions = [edge_list[i:i + chunk_size] for i in range(0, len(edge_list), chunk_size)]

    with Pool(processes=num_workers) as pool:
        if method == "merge":
            results = pool.starmap(tc_merge_worker, [(chunk, A_plus) for chunk in partitions])
        elif method == "hash":
            A_sets = build_A_sets(A_plus)
            results = pool.starmap(tc_hash_worker, [(chunk, A_sets) for chunk in partitions])
        else:
            raise ValueError("Invalid method. Use 'merge' or 'hash'.")

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
    filename = "Email-Enron.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        for method in ["merge", "hash"]:
            print(f"\n===== Running TC-{method.upper()} =====")
            start_time = time.time()
            total_triangles = parallel_triangle_count(graph, 4, method=method)
            end_time = time.time()
            print(f"Total triangles ({method}): {total_triangles}")
            print(f"Triangle Algorithm time ({method}): {end_time - start_time:.4f} seconds")

        print("\n===== NetworkX Verification =====")
        start_time = time.time()
        triangles_networkx = nx.triangles(graph)
        triangles_networkx_count = sum(triangles_networkx.values()) // 3
        end_time = time.time()
        print(f"Total triangles (NetworkX): {triangles_networkx_count}")
        print(f"Triangle Algorithm time (NetworkX): {end_time - start_time:.4f} seconds")

    except FileNotFoundError:
        print("File not found")
