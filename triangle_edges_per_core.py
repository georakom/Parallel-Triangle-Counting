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
        A_plus[node].sort()  # Needed for merge-based intersection
    return A_plus

def tc_merge_worker(edges, A_plus):
    from bisect import bisect_left
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(edges)} edges")
    
    triangle_count = 0
    for u, v in edges:
        neighbors_u = A_plus[u]
        neighbors_v = A_plus[v]
        # Merge step (both lists are sorted)
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

    print(f"[Worker {pid}] FINISHED. Found {triangle_count} triangles.", flush = True)
    return triangle_count


def parallel_triangle_count(G, num_workers):
    # Build node ranking
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    node_ranking = {node: i for i, node in enumerate(nodes_sorted_by_degree)}
    
    A_plus = build_A_plus(G, node_ranking)

    # Build edge list from A⁺
    edge_list = [(u, v) for u in A_plus for v in A_plus[u]]
    
    # Split edges into chunks by dividing the total number of edges to the number of cores
    chunk_size = len(edge_list) // num_workers
    partitions = [edge_list[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers-1)] + [edge_list[(num_workers-1)*chunk_size:]]

    # Using smaller edge chunks, Open-MP style
    #chunk_size = 5000
    #partitions = [edge_list[i:i + chunk_size] for i in range(0, len(edge_list), chunk_size)]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(tc_merge_worker, [(chunk, A_plus) for chunk in partitions])

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


        