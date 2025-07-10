import time
from multiprocessing import Pool, get_context
import numpy as np
import scipy.sparse as sp
from numba import njit
import gc
import Partitioners as p

"""
Pass Arrays, not CSR Objects - the main idea - best implementation till now
Optimized for faster preprocessing by combining ordering and partitioning
"""
def read_graph_to_csr(filename):
    edges = []
    node_set = set()
    with open(filename, 'r') as f:
        for line in f:
            try:
                u, v = map(int, line.strip().split())
                if u == v:
                    continue
                edges.append((u, v))
                node_set.add(u)
                node_set.add(v)
            except:
                continue
    nodes = sorted(node_set)
    node_id_map = {node: i for i, node in enumerate(nodes)}
    remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in edges]
    rows, cols = zip(*remapped_edges)
    data = np.ones(len(rows), dtype=np.uint8)
    N = len(nodes)
    adj_upper = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj_upper + adj_upper.T  # undirected
    adj = adj.tocsr()
    return adj, nodes, node_id_map

def extract_subgraph_for_worker(adj, master_nodes, node_to_order):
    nodes_needed = set(master_nodes)
    for u in master_nodes:
        row_start, row_end = adj.indptr[u], adj.indptr[u + 1]
        neighbors = adj.indices[row_start:row_end]
        u_order = node_to_order[u]
        for v in neighbors:
            if node_to_order[v] > u_order:
                nodes_needed.add(v)
    nodes_needed = sorted(nodes_needed)
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(nodes_needed)}
    sub_adj = adj[nodes_needed, :][:, nodes_needed].tocsr()
    master_local_ids = {global_to_local[u] for u in master_nodes}
    local_to_global = {local_id: global_id for global_id, local_id in global_to_local.items()}
    return sub_adj, master_local_ids, global_to_local, local_to_global


def extract_all_worker_data(adj, partitions, num_workers, node_to_order):
    worker_data = []
    for wid in range(num_workers):
        master_nodes = set(partitions[wid])
        sub_adj, master_local_ids, global_to_local, local_to_global = extract_subgraph_for_worker(adj, master_nodes, node_to_order)
        num_mirrors = len(global_to_local) - len(master_nodes)
        print(f"Worker {wid} has {num_mirrors} mirror nodes.")
        indptr = sub_adj.indptr
        indices = sub_adj.indices
        n = sub_adj.shape[0]
        master_mask = np.zeros(n, dtype=np.bool_)
        for idx in master_local_ids:
            master_mask[idx] = True
        order_array = node_to_order.copy()
        local_to_global_array = np.empty(n, dtype=np.int32)
        for local, global_id in local_to_global.items():
            local_to_global_array[local] = global_id
        worker_data.append((indptr, indices, master_mask, order_array, local_to_global_array))
    return worker_data

@njit
def count_triangles_master_only(indptr, indices, master_mask, order_array, local_to_global_array):
    count = 0
    n = len(local_to_global_array)
    for u_local in range(n):
        if not master_mask[u_local]:
            continue
        u_global = local_to_global_array[u_local]
        u_order = order_array[u_global]
        u_neighbors = indices[indptr[u_local]:indptr[u_local + 1]]
        for v_local in u_neighbors:
            v_global = local_to_global_array[v_local]
            v_order = order_array[v_global]
            if v_order <= u_order:
                continue
            v_neighbors = indices[indptr[v_local]:indptr[v_local + 1]]
            i = 0
            j = 0
            while i < len(u_neighbors) and j < len(v_neighbors):
                w_local = u_neighbors[i]
                w_global = local_to_global_array[w_local]
                w_order = order_array[w_global]
                if w_local == v_local or w_order <= v_order:
                    i += 1
                    continue
                if w_local == v_neighbors[j]:
                    count += 1
                    i += 1
                    j += 1
                elif w_local < v_neighbors[j]:
                    i += 1
                else:
                    j += 1
    return count

def count_triangles_worker(args):
    return count_triangles_master_only(*args)


def parallel_triangle_count(graph_csr, num_workers):
    partition_time = time.time()
    partitions, assignments, node_to_order = p.neighbor_metis_partition(graph_csr, num_workers)
    print(f"Partitioning + ordering took {time.time() - partition_time:.2f} seconds")

    data_prep_time = time.time()
    worker_data = extract_all_worker_data(graph_csr, partitions, num_workers, node_to_order)
    print(f"Data extraction took {time.time() - data_prep_time:.2f} seconds")
    del partitions
    del node_to_order
    del graph_csr
    gc.collect()
    triangle_count_time = time.time()
    with get_context("fork").Pool(num_workers) as pool:
        results = pool.map(count_triangles_worker, worker_data)
    print(f"Triangle counting took {time.time() - triangle_count_time:.2f} seconds")

    for wid, count in enumerate(results):
        print(f"Worker {wid} counted {count} triangles")
    return sum(results)


if __name__ == "__main__":
    filepath = "/data/delab/georakom/"
    filename = "as-skitter.txt"
    num_workers = 4
    try:
        graph_csr, nodes, node_id_map = read_graph_to_csr(filepath + filename)
        del nodes
        del node_id_map
        gc.collect()
        start_time = time.time()
        total_triangles = parallel_triangle_count(graph_csr, num_workers)
        end_time = time.time()
        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)
    except FileNotFoundError:
        print("Graph file not found.")
