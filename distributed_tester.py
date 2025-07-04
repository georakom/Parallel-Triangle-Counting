import time
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from numba import njit

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
    node_id_map = {node: i for i, node in enumerate(nodes)}  # original id -> compact id
    remapped_edges = [(node_id_map[u], node_id_map[v]) for u, v in edges]
    rows, cols = zip(*remapped_edges)
    data = np.ones(len(rows), dtype=np.uint8)
    N = len(nodes)
    adj_upper = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    adj = adj_upper + adj_upper.T
    #adj.sort_indices()
    adj = adj.tocsr()
    return adj, nodes, node_id_map

def get_ordering(adj):
    degrees = np.array(adj.sum(axis=1)).flatten()
    order = np.lexsort((np.arange(adj.shape[0]), degrees))
    node_to_order = {node: rank for rank, node in enumerate(order)}
    order_to_node = {rank: node for rank, node in enumerate(order)}
    return node_to_order, order_to_node

def partition_by_master(adj, num_workers):
    node_to_order, order_to_node = get_ordering(adj)
    n = adj.shape[0]
    # Assign master by order
    order_to_host = {rank: rank % num_workers for rank in range(n)}
    partitions = [[] for _ in range(num_workers)]
    assignments = np.zeros(n, dtype=int)
    for node in range(n):
        host = order_to_host[node_to_order[node]]
        partitions[host].append(node)
        assignments[node] = host
    return partitions, assignments, node_to_order, order_to_node

def extract_subgraph_for_worker(adj, master_nodes):
    # For each master node, include all its neighbors (mirrors)
    nodes_needed = set(master_nodes)
    for u in master_nodes:
        row_start, row_end = adj.indptr[u], adj.indptr[u + 1]
        neighbors = adj.indices[row_start:row_end]
        nodes_needed.update(neighbors)
    nodes_needed = sorted(nodes_needed)
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(nodes_needed)}
    sub_adj = adj[nodes_needed, :][:, nodes_needed].tocsr()
    master_local_ids = {global_to_local[u] for u in master_nodes}
    local_to_global = {local_id: global_id for global_id, local_id in global_to_local.items()}
    return sub_adj, master_local_ids, global_to_local, local_to_global

def extract_all_worker_data_master_mirror(adj, partitions, num_workers, node_to_order):
    worker_data = []
    for wid in range(num_workers):
        master_nodes = set(partitions[wid])
        sub_adj, master_local_ids, global_to_local, local_to_global = extract_subgraph_for_worker(adj, master_nodes)
        worker_data.append((sub_adj, master_local_ids, node_to_order, local_to_global))
    return worker_data

@njit
def count_triangles_master_only(csr_matrix_indptr, csr_matrix_indices, master_nodes, order_array, local_to_global_array):
    count = 0
    n = len(local_to_global_array)
    for u_local in range(n):
        if not master_nodes[u_local]:
            continue
        u_global = local_to_global_array[u_local]
        u_order = order_array[u_global]
        u_neighbors = csr_matrix_indices[csr_matrix_indptr[u_local]:csr_matrix_indptr[u_local + 1]]
        for v_local in u_neighbors:
            v_global = local_to_global_array[v_local]
            v_order = order_array[v_global]
            if v_order <= u_order:
                continue
            v_neighbors = csr_matrix_indices[csr_matrix_indptr[v_local]:csr_matrix_indptr[v_local + 1]]
            # Intersection for w > v
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

def count_triangles_worker_master_mirror(args):
    csr_matrix, master_local_ids, node_to_order, local_to_global = args
    n = csr_matrix.shape[0]
    master_mask = np.zeros(n, dtype=np.bool_)
    for idx in master_local_ids:
        master_mask[idx] = True
    # order_array: compact node id -> order
    # node_to_order is {compact (0..N-1): order}
    N = max(node_to_order.keys()) + 1
    order_array = np.zeros(N, dtype=np.int32)
    for node, order in node_to_order.items():
        order_array[node] = order
    # local_to_global_array: local id -> compact id
    local_to_global_array = np.zeros(n, dtype=np.int32)
    for local, global_id in local_to_global.items():
        local_to_global_array[local] = global_id
    return count_triangles_master_only(
        csr_matrix.indptr, csr_matrix.indices, master_mask, order_array, local_to_global_array
    )

def parallel_triangle_count_master_mirror(graph_csr, num_workers, partition_func):
    partitions, assignments, node_to_order, order_to_node = partition_func(graph_csr, num_workers)
    worker_data = extract_all_worker_data_master_mirror(graph_csr, partitions, num_workers, node_to_order)
    with Pool(num_workers) as pool:
        results = pool.map(count_triangles_worker_master_mirror, worker_data)
    for wid, count in enumerate(results):
        print(f"Worker {wid} counted {count} triangles")
    return sum(results)

if __name__ == "__main__":
    filepath = "/data/delab/georakom/"
    filename = "com-lj.ungraph.txt"
    num_workers = 4
    try:
        graph_csr, nodes, node_id_map = read_graph_to_csr(filepath + filename)
        start_time = time.time()
        total_triangles = parallel_triangle_count_master_mirror(
            graph_csr, num_workers, partition_by_master
        )
        end_time = time.time()
        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)
    except FileNotFoundError:
        print("Graph file not found.")