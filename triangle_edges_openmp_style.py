import networkx as nx
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from collections import defaultdict
import random
import time
import os

# Shared-Memory Parallel Triangle Counting is implemented with TC-Merge and TC-Hash
# Work between cores is split based on a specific number (for example 5000 edges) for a better balance

def read_graph_from_file(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    random.shuffle(edges)
    for u, v in edges:
        G.add_edge(u, v)
    return G


def rank_by_degree(G):
    nodes_sorted = sorted(G.nodes(), key=lambda x: G.degree[x])
    rank = {node: i for i, node in enumerate(nodes_sorted)}
    return rank


def build_A_plus_csr(G, rank):
    A_plus = defaultdict(list)
    for v in G.nodes():
        for w in G.neighbors(v):
            if rank[v] < rank[w]:
                A_plus[v].append(w)
    for v in A_plus:
        A_plus[v].sort()
    nodes = sorted(A_plus.keys())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    indptr = [0]
    indices = []
    for v in nodes:
        neighbors = A_plus[v]
        indices.extend(neighbors)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int64), np.array(indices, dtype=np.int64), nodes, node_to_idx


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


def hash_intersect_count(arr_small, arr_large_set):
    return sum(1 for x in arr_small if x in arr_large_set)


def worker_merge(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(nodes_chunk)} nodes (MERGE)")
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

    print(f"[Worker {pid}] FINISHED. Found {local_count} triangles.", flush=True)
    return_dict[mp.current_process().name] = local_count


def worker_hash(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict):
    pid = os.getpid()
    print(f"[Worker {pid}] STARTED with {len(nodes_chunk)} nodes (HASH)")

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
            # Use smaller array to query larger set for hashing
            if len(neighbors_v) < len(neighbors_w):
                local_count += hash_intersect_count(neighbors_v, set(neighbors_w))
            else:
                local_count += hash_intersect_count(neighbors_w, set(neighbors_v))
    shm_indptr.close()
    shm_indices.close()

    print(f"[Worker {pid}] FINISHED. Found {local_count} triangles.", flush=True)
    return_dict[mp.current_process().name] = local_count

def _worker_wrapper(method, shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict, sem):
    try:
        if method == "merge":
            worker_merge(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict)
        elif method == "hash":
            worker_hash(shm_name_indptr, shm_name_indices, n_nodes, nodes_chunk, node_to_idx, return_dict)
        else:
            raise ValueError("Invalid method")
    finally:
        sem.release()


def parallel_triangle_count(G, num_workers=4, method="merge", chunk_size=10000):
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)

    # Shared memory setup
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)[:] = indptr[:]
    np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)[:] = indices[:]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    sem = mp.Semaphore(num_workers)

    # Schedule smaller dynamic chunks like OpenMP dynamic schedule
    for i in range(0, len(nodes), chunk_size):
        nodes_chunk = nodes[i:i + chunk_size]
        sem.acquire()
        p = mp.Process(target=_worker_wrapper, args=(
            method,
            shm_indptr.name,
            shm_indices.name,
            len(nodes),
            nodes_chunk,
            node_to_idx,
            return_dict,
            sem
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_triangles = sum(return_dict.values())

    shm_indptr.close()
    shm_indptr.unlink()
    shm_indices.close()
    shm_indices.unlink()

    return total_triangles




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
