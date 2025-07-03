import time
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from numba import njit

def extract_all_worker_data(adj, partitions, assignments, num_workers):
    start = time.time()
    worker_data = []

    for wid in range(num_workers):
        master_nodes = set(partitions[wid])
        row_mask = np.zeros(adj.shape[0], dtype=bool)

        # Include all rows that either belong to master_nodes or are neighbors
        for u in master_nodes:
            row_mask[u] = True
            row_start, row_end = adj.indptr[u], adj.indptr[u + 1]
            neighbors = adj.indices[row_start:row_end]
            row_mask[neighbors] = True  # Include mirrors

        sub_adj = adj[row_mask][:, row_mask]
        sub_adj.sort_indices()

        global_to_local = {global_id: local_id for local_id, global_id in enumerate(np.flatnonzero(row_mask))}
        master_local_ids = {global_to_local[u] for u in master_nodes}

        worker_data.append((sub_adj, master_local_ids))
    print(f"Worker data extraction took: {time.time() - start:.4f} seconds")
    return worker_data


def count_triangles_worker(args):
    csr_matrix, master_nodes = args
    count = 0

    for u in master_nodes:
        neighbors_u = csr_matrix.indices[csr_matrix.indptr[u]:csr_matrix.indptr[u + 1]]
        for v in neighbors_u:
            if u >= v:
                continue
            neighbors_v = csr_matrix.indices[csr_matrix.indptr[v]:csr_matrix.indptr[v + 1]]
            count += merge_intersection_count(neighbors_u, neighbors_v)
    print(f"Worker found: {count} triangles", flush = True)
    return count

@njit
def merge_intersection_count(a, b):
    count = 0
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            count += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return count


def parallel_triangle_count(graph_csr, num_workers, partition_func):
    start = time.time()
    partitions, assignments = partition_func(graph_csr, num_workers)
    prep_start = time.time()

    worker_data = extract_all_worker_data(graph_csr, partitions, assignments, num_workers)
    print(f"Data preparation took: {time.time() - prep_start:.4f} seconds")
    print(f"Preprocessing took: {time.time() - start:.4f} seconds")

    triangle_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(count_triangles_worker, worker_data)
    print(f"Pure triangle counting took: {time.time() - triangle_time:.4f} seconds")
    return sum(results) // 3


def read_graph_to_csr(filename):
    edges = []
    node_set = set()

    with open(filename, 'r') as f:
        for line in f:
            try:
                u, v = map(int, line.strip().split())
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                edges.append((u, v))
                node_set.add(u)
                node_set.add(v)
            except:
                continue

    nodes = sorted(node_set)
    node_id_map = {node: i for i, node in enumerate(nodes)}  # Map original IDs to 0...N-1
    remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in edges]

    rows, cols = zip(*remapped_edges)
    data = np.ones(len(rows), dtype=np.uint8)
    N = len(nodes)

    adj_upper = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj_upper + adj_upper.T  # Make it symmetric
    adj = adj.tocsr()

    return adj, nodes, node_id_map

if __name__ == "__main__":
    import Partitioners as p # Importing the partitioning algorithm to use

    filepath = "/data/delab/georakom/"
    filename = "com-lj.ungraph.txt"

    try:
        graph_csr, nodes, node_id_map = read_graph_to_csr(filepath + filename)

        start_time = time.time()
        total_triangles = parallel_triangle_count(graph_csr, 4, p.hashing_partition) # Deciding the partition
        end_time = time.time()

        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)

    except FileNotFoundError:
        print("Graph file not found.")

