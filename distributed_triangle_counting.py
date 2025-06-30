import random
import time
from multiprocessing import Pool
import networkx as nx
from collections import defaultdict
import psutil
import os

def extract_all_worker_data(G, partitions, assignments, num_workers):
    # Per worker storage
    worker_edges = [set() for _ in range(num_workers)]
    worker_nodes_used = [set() for _ in range(num_workers)]
    node_to_workers = defaultdict(set)

    # Fast adjacency lookup to avoid second edge pass
    edge_set = set()
    for u, v in G.edges():
        if u > v:
            u, v = v, u
        edge_set.add((u, v))  # store directed once
        worker_id = assignments[u]
        worker_edges[worker_id].add((u, v))
        worker_nodes_used[worker_id].add(u)
        worker_nodes_used[worker_id].add(v)
        node_to_workers[u].add(worker_id)
        node_to_workers[v].add(worker_id)

    # Efficient inline proxy edge insertion: resolve only needed duplicates
    for u, v in edge_set:
        shared = node_to_workers[u] & node_to_workers[v]
        for wid in shared:
            if (u, v) not in worker_edges[wid]:
                worker_edges[wid].add((u, v))
                worker_nodes_used[wid].add(u)
                worker_nodes_used[wid].add(v)

    # Final prep
    worker_data = []
    for wid in range(num_workers):
        masters = set(partitions[wid])
        used = worker_nodes_used[wid]
        mirrors = used.difference(masters)
        print(f"[Worker {wid}] Mirror nodes: {len(mirrors)}")
        worker_data.append((list(worker_edges[wid]), masters))

    return worker_data

# def extract_all_worker_data(G, partitions, assignments, num_workers):
#     # Initialize storage per worker
#     worker_edges = [set() for _ in range(num_workers)]
#     worker_nodes_used = [set() for _ in range(num_workers)]
#
#     # Track which workers need each node
#     node_to_workers = defaultdict(set)
#
#     # Pass 1: assign directed edge (u → v) to master of u, and update who uses what
#     for u, v in G.edges():
#         if u > v:
#             u, v = v, u  # enforce direction u → v
#
#         worker_id = assignments[u]
#         worker_edges[worker_id].add((u, v))
#         worker_nodes_used[worker_id].update([u, v])
#         node_to_workers[u].add(worker_id)
#         node_to_workers[v].add(worker_id)
#
#     # Inline proxy edge insertion
#     for u, v in G.edges():
#         if u > v:
#             u, v = v, u
#
#         shared_workers = node_to_workers[u] & node_to_workers[v]
#         for wid in shared_workers:
#             worker_edges[wid].add((u, v))
#
#     # Final prep
#     worker_data = []
#     for worker_id in range(num_workers):
#         masters = set(partitions[worker_id])
#         used_nodes = {u for edge in worker_edges[worker_id] for u in edge}
#         mirrors = used_nodes - masters
#         print(f"[Worker {worker_id}] Mirror nodes: {len(mirrors)}")
#         worker_data.append((list(worker_edges[worker_id]), masters))
#
#     return worker_data

def count_triangles(edge_list, master_nodes):
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
        results = pool.starmap(count_triangles, worker_data)
    print(f"Pure triangle counting took: {time.time() - triangle_time:.4f} seconds")
    return sum(results)

def read_graph_from_file(filename, batch_size=1_000_000):
    G = nx.Graph()
    edge_buffer = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    u, v = map(int, parts)
                    edge_buffer.append((u, v))

                    # Process in batches to limit memory
                    if len(edge_buffer) >= batch_size:
                        random.shuffle(edge_buffer)
                        G.add_edges_from(edge_buffer)
                        edge_buffer = []
                except ValueError:
                    continue

        # Add remaining edges
        if edge_buffer:
            random.shuffle(edge_buffer)
            G.add_edges_from(edge_buffer)

    return G

if __name__ == "__main__":
    import Partitioners as p # Importing the partitioning algorithm to use

    filepath = "./data/"
    filename = "com-youtube.ungraph.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        start_time = time.time()
        total_triangles = parallel_triangle_count(graph, 4, p.ldg_partition) # Deciding the partition
        end_time = time.time()

        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)

    except FileNotFoundError:
        print("Graph file not found.")