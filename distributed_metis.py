import networkx as nx
import os
from multiprocessing import Pool
import time
import random
from collections import defaultdict
import metis

def preprocess_graph(G):
    """Builds DAG and computes node ranking based on degree."""
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    node_ranking = {node: i for i, node in enumerate(nodes_sorted_by_degree)}
    DAG = nx.DiGraph()

    for u, v in G.edges():
        if node_ranking[u] < node_ranking[v]:
            DAG.add_edge(u, v)
        else:
            DAG.add_edge(v, u)

    return DAG, node_ranking


def build_proxy_subgraph(worker_id, master_nodes, all_edges, node_ranking, worker_assignments):
    # Step 1: Collect all edges assigned to this worker
    local_edges = [e for e in all_edges if worker_assignments[e[0]] == worker_id]

    # Step 2: Identify all nodes needed
    local_nodes = set()
    for u, v in local_edges:
        local_nodes.add(u)  # master
        local_nodes.add(v)  # potential mirror

    # Step 3: Include edge proxies (triangle-completing edges)
    # Add any edge from DAG if both u and v are in local_nodes
    full_local_edges = set(local_edges)
    for u, v in all_edges:
        if u in local_nodes and v in local_nodes:
            full_local_edges.add((u, v))  # ensure triangle closure

    # Step 4: Build subgraph
    subgraph = nx.DiGraph()
    # Add all master nodes explicitly (even if they have no edges)
    subgraph.add_nodes_from(set(master_nodes) | local_nodes)
    subgraph.add_edges_from(full_local_edges)

    return subgraph, {
        "master_nodes": set(master_nodes),
        "node_ranking": node_ranking
    }


def triangle_count_worker(G_local, master_nodes, node_ranking):
    count = 0
    for v in master_nodes:
        neighbors_v = {u for u in G_local.successors(v) if node_ranking[u] > node_ranking[v]}
        for u in neighbors_v:
            neighbors_u = {w for w in G_local.successors(u) if node_ranking[w] > node_ranking[u]}
            count += len(neighbors_v & neighbors_u)
    print(f"Worker FINISHED. Found {count} triangles.", flush=True)
    return count


def parallel_triangle_count(G, num_workers):
    start_pre_time = time.time()

    DAG, node_ranking = preprocess_graph(G)

    # After DAG is built
    worker_assignments = {}  # node -> worker
    partitions = [[] for _ in range(num_workers)]

    # Use METIS to assign master nodes (just node -> partition)
    metis_start = time.time()
    _, parts = metis.part_graph(DAG, nparts=num_workers)
    metis_finish = time.time()
    print(f"METIS partition alone took: {metis_finish - metis_start:.4f} seconds.")
    for node, part in zip(G.nodes(), parts):
        worker_assignments[node] = part
        partitions[part].append(node)

    # Now assign all outgoing edges of each master to the same worker
    worker_edges = [set() for _ in range(num_workers)]
    for u, v in DAG.edges():
        master_worker = worker_assignments[u]  # Because u → v
        worker_edges[master_worker].add((u, v))

    proxy_subgraphs = []

    for worker_id in range(num_workers):
        master_nodes = partitions[worker_id]
        subgraph, meta = build_proxy_subgraph(
            worker_id=worker_id,
            master_nodes=master_nodes,
            all_edges=DAG.edges(),
            node_ranking=node_ranking,
            worker_assignments=worker_assignments
        )
        proxy_subgraphs.append((subgraph, meta))

    end_pre_time = time.time()
    print(f"Preprocessing took: {end_pre_time - start_pre_time:.6f} seconds")

    triangle_time = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            triangle_count_worker,
            [(subgraph, meta["master_nodes"], meta["node_ranking"]) for subgraph, meta in proxy_subgraphs]
        )
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