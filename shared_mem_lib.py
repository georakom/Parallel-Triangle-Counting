import time
import networkx as nx
import numpy as np
import random
from multiprocessing import shared_memory
import multiprocessing as mp
import os

# Assigns a total order (ranking) to nodes based on degree to build the A+ representation
def rank_by_degree(G):
    node_list = np.array(G.nodes(), dtype=np.int64)
    degrees = np.array([G.degree[n] for n in node_list])
    sorted_indices = np.argsort(degrees)  # ascending by degree
    rank = {node: i for i, node in enumerate(node_list[sorted_indices])}
    return rank

# Builds the A+ CSR (Compressed Sparse Row) structure for efficient triangle counting
def build_A_plus_csr(G, rank):
    node_list = np.array(G.nodes(), dtype=np.int64)
    node_idx_map = {node: i for i, node in enumerate(node_list)}
    num_nodes = len(node_list)

    # Preallocate large enough space; we’ll trim later
    indices_buffer = np.empty(G.number_of_edges(), dtype=np.int64)  # overallocate for safety CHANGED IT TO TEST
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)

    edge_ptr = 0
    A_plus_counts = np.zeros(num_nodes, dtype=np.int64)

    # Fill A+ structure: only store neighbors with higher rank
    for v in node_list:
        i = node_idx_map[v]
        count = 0
        for w in G.neighbors(v):
            if rank[v] < rank[w]:
                indices_buffer[edge_ptr] = w
                edge_ptr += 1
                count += 1
        A_plus_counts[i] = count

    # Build indptr from counts
    np.cumsum(A_plus_counts, out=indptr[1:])

    # Trim indices to actual size
    indices = indices_buffer[:edge_ptr]

    # Filter out nodes with zero A⁺ neighbors (they won’t contribute triangles)
    non_empty = A_plus_counts > 0
    nodes = node_list[non_empty]
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Rebuild compacted indptr/indices for only these nodes
    new_indptr = [0]
    new_indices = []

    for node in nodes:
        i = node_idx_map[node]
        start = indptr[i]
        end = indptr[i + 1]
        neighbors = np.sort(indices[start:end])  # Still sort for merge
        new_indices.extend(neighbors)
        new_indptr.append(len(new_indices))

    return (
        np.array(new_indptr, dtype=np.int64),
        np.array(new_indices, dtype=np.int64),
        list(nodes),
        node_to_idx,
    )

# Intersect two sorted arrays (neighbors) using the merge-based approach
def merge_intersect_count(arr1, arr2):
    count = 0
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] == arr2[j]:
            count += 1
            i += 1
            j += 1
        elif arr1[i] < arr2[j]:
            i += 1
        else:
            j += 1
    return count

# Intersect two neighbor arrays using hash-based lookup
def hash_intersect_count(arr_small, arr_large_set):
    return sum(1 for x in arr_small if x in arr_large_set) # One side is converted to a set; the other is scanned

# Worker for merge-based triangle counting
def worker_merge(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(nodes_chunk)} nodes (MERGE)")
    tri_time = time.time()

    # Reconnect to shared memory
    shm_indptr = shared_memory.SharedMemory(name=shm_name_indptr)
    shm_indices = shared_memory.SharedMemory(name=shm_name_indices)
    indptr = np.ndarray((n_nodes + 1,), dtype=np.int64, buffer=shm_indptr.buf)
    indices = np.ndarray((indptr[-1],), dtype=np.int64, buffer=shm_indices.buf)

    local_count = 0
    for v in nodes_chunk:
        v_idx = node_to_idx[v]
        neighbors_v = indices[indptr[v_idx]:indptr[v_idx + 1]]
        for w in neighbors_v:
            if w not in node_to_idx:
                continue
            w_idx = node_to_idx[w]
            neighbors_w = indices[indptr[w_idx]:indptr[w_idx + 1]]
            local_count += merge_intersect_count(neighbors_v, neighbors_w)
    shm_indptr.close()
    shm_indices.close()
    print(f"[Worker {pid}] FINISHED. Found {local_count} triangles, in {time.time() - tri_time:.4f} secs", flush=True)
    return_dict[mp.current_process().name] = local_count

# Worker for hash-based triangle counting
def worker_hash(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(nodes_chunk)} nodes (HASH)")
    tri_time = time.time()

    shm_indptr = shared_memory.SharedMemory(name=shm_name_indptr)
    shm_indices = shared_memory.SharedMemory(name=shm_name_indices)
    indptr = np.ndarray((n_nodes + 1,), dtype=np.int64, buffer=shm_indptr.buf)
    indices = np.ndarray((indptr[-1],), dtype=np.int64, buffer=shm_indices.buf)

    local_count = 0
    for v in nodes_chunk:
        v_idx = node_to_idx[v]
        neighbors_v = indices[indptr[v_idx]:indptr[v_idx + 1]]
        for w in neighbors_v:
            if w not in node_to_idx:
                continue
            w_idx = node_to_idx[w]
            neighbors_w = indices[indptr[w_idx]:indptr[w_idx + 1]]

            # Pick smaller list to probe through larger set
            if len(neighbors_v) < len(neighbors_w):
                local_count += hash_intersect_count(neighbors_v, set(neighbors_w))
            else:
                local_count += hash_intersect_count(neighbors_w, set(neighbors_v))
    shm_indptr.close()
    shm_indices.close()

    print(f"[Worker {pid}] FINISHED. Found {local_count} triangles in {time.time() - tri_time:.4f} secs.", flush=True)
    return_dict[mp.current_process().name] = local_count

# Wrapper to allow releasing semaphore after work finishes
def worker_wrapper(method, shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict, sem):
    try:
        if method == "merge":
            worker_merge(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict)
        elif method == "hash":
            worker_hash(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict)
        else:
            raise ValueError("Invalid method")
    finally:
        sem.release()

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