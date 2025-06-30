import random
import time
from multiprocessing import Pool
import networkx as nx
from collections import defaultdict
import psutil
import os
from numba import njit
import numpy as np

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

def convert_graph_to_edge_array(G):
    """Convert NetworkX graph to a NumPy edge array (u < v enforced)."""
    edge_list = [(min(u, v), max(u, v)) for u, v in G.edges()]
    return np.array(edge_list, dtype=np.int32)

@njit
def compute_worker_data_numba(edges, assignments_arr, node_worker_mask, num_workers, num_nodes):
    # Preallocate space: estimate upper bound (to be trimmed later)
    max_edges_per_worker = len(edges) * 2  # very safe upper bound
    edge_buffers = [np.empty((max_edges_per_worker, 2), dtype=np.int32) for _ in range(num_workers)]
    edge_counts = np.zeros(num_workers, dtype=np.int32)

    node_usage = np.zeros((num_nodes, num_workers), dtype=np.uint8)

    for i in range(len(edges)):
        u, v = edges[i]
        uid = assignments_arr[u]
        edge_buffers[uid][edge_counts[uid]] = [u, v]
        edge_counts[uid] += 1

        node_usage[u, uid] = 1
        node_usage[v, uid] = 1

    # Inline proxy edge logic
    for i in range(len(edges)):
        u, v = edges[i]
        for wid in range(num_workers):
            if node_usage[u, wid] and node_usage[v, wid]:
                edge_buffers[wid][edge_counts[wid]] = [u, v]
                edge_counts[wid] += 1

    # Trim and return
    result = []
    for wid in range(num_workers):
        result.append(edge_buffers[wid][:edge_counts[wid]])
    return result


def extract_all_worker_data(G, partitions, assignments, num_workers):
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    num_nodes = len(G)

    edge_array = convert_graph_to_edge_array(G)
    assignments_arr = np.zeros(num_nodes, dtype=np.int32)

    for node, worker in assignments.items():
        assignments_arr[node_to_index[node]] = worker

    # Used in Numba: each row (node) shows which workers use it
    node_worker_mask = np.zeros((num_nodes, num_workers), dtype=np.uint8)

    # Compute with Numba
    raw_worker_edges = compute_worker_data_numba(
        np.array([[node_to_index[u], node_to_index[v]] for u, v in edge_array], dtype=np.int32),
        assignments_arr,
        node_worker_mask,
        num_workers,
        num_nodes
    )

    # Convert back to original node labels
    worker_data = []
    for wid in range(num_workers):
        edges = [(index_to_node[u], index_to_node[v]) for u, v in raw_worker_edges[wid]]
        masters = set(partitions[wid])
        used_nodes = {u for e in edges for u in e}
        mirrors = used_nodes - masters
        print(f"[Worker {wid}] Mirror nodes: {len(mirrors)}")
        worker_data.append((edges, masters))

    return worker_data



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
    filename = "amazon.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        start_time = time.time()
        total_triangles = parallel_triangle_count(graph, 4, p.ldg_partition) # Deciding the partition
        end_time = time.time()

        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)

        # start_time = time.time()
        # triangles_networkx = nx.triangles(graph).values() // 3
        # end_time = time.time()
        # print(f"Total triangles (NetworkX): {triangles_networkx}")
        # print("Triangle Algorithm time (NetworkX): ", end_time - start_time)

    except FileNotFoundError:
        print("Graph file not found.")