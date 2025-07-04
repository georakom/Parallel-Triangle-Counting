import time
from multiprocessing import Pool
import numpy as np
from numba import njit

def read_graph_to_dict(filename):
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
    adj = {i: set() for i in range(len(nodes))}
    for u, v in edges:
        u_idx = node_id_map[u]
        v_idx = node_id_map[v]
        adj[u_idx].add(v_idx)
        adj[v_idx].add(u_idx)
    return adj, nodes, node_id_map

def get_ordering(adj):
    degrees = np.array([len(adj[u]) for u in sorted(adj.keys())])
    order = np.lexsort((np.arange(len(adj)), degrees))
    node_to_order = {node: rank for rank, node in enumerate(order)}
    order_to_node = {rank: node for rank, node in enumerate(order)}
    return node_to_order, order_to_node

def partition_by_master(adj, num_workers):
    node_to_order, order_to_node = get_ordering(adj)
    n = len(adj)
    order_to_host = {rank: rank % num_workers for rank in range(n)}
    partitions = [[] for _ in range(num_workers)]
    assignments = np.zeros(n, dtype=int)
    for node in range(n):
        host = order_to_host[node_to_order[node]]
        partitions[host].append(node)
        assignments[node] = host
    return partitions, assignments, node_to_order, order_to_node

def extract_subgraph_for_worker(adj, master_nodes):
    nodes_needed = set(master_nodes)
    for u in master_nodes:
        nodes_needed.update(adj[u])
    nodes_needed = sorted(nodes_needed)
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(nodes_needed)}
    # Build subgraph as dict-of-sets
    sub_adj = {}
    for global_id in nodes_needed:
        local_id = global_to_local[global_id]
        sub_adj[local_id] = set(global_to_local[v] for v in adj[global_id] if v in global_to_local)
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

def dict_subgraph_to_csr(sub_adj, n):
    indptr = [0]
    indices = []
    for u in range(n):
        neighbors = sorted(sub_adj[u])
        indices.extend(neighbors)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32)

@njit
def count_triangles_master_only_csr(indptr, indices, master_mask, order_array, local_to_global_array):
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
            # Merge-based intersection for w > v
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
    build_time = time.time()
    sub_adj, master_local_ids, node_to_order, local_to_global = args
    n = len(sub_adj)
    # Build CSR
    indptr, indices = dict_subgraph_to_csr(sub_adj, n)
    # Prepare masks and order arrays
    max_global_id = max(node_to_order.keys())
    order_array = np.zeros(max_global_id + 1, dtype=np.int32)
    for node, order in node_to_order.items():
        order_array[node] = order
    local_to_global_array = np.zeros(n, dtype=np.int32)
    for local, global_id in local_to_global.items():
        local_to_global_array[local] = global_id
    master_mask = np.zeros(n, dtype=bool)
    for idx in master_local_ids:
        master_mask[idx] = True
    print(f"Worker built CSR in {time.time() - build_time:.2f} seconds")
    triangle_time = time.time()
    count = count_triangles_master_only_csr(
        indptr, indices, master_mask, order_array, local_to_global_array
    )
    print(f"Worker found the triangles in {time.time() - triangle_time:.2f} seconds")
    return count

def parallel_triangle_count_master_mirror(graph_adj, num_workers, partition_func):
    partition_time = time.time()
    partitions, assignments, node_to_order, order_to_node = partition_func(graph_adj, num_workers)
    print(f"Partitioning took {time.time() - partition_time:.2f} seconds")

    data_prep_time = time.time()
    worker_data = extract_all_worker_data_master_mirror(graph_adj, partitions, num_workers, node_to_order)
    print(f"Data extraction took {time.time() - data_prep_time:.2f} seconds")

    triangle_count_time = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(count_triangles_worker_master_mirror, worker_data)
    print(f"Triangle counting took {time.time() - triangle_count_time:.2f} seconds")

    for wid, count in enumerate(results):
        print(f"Worker {wid} counted {count} triangles")
    return sum(results)

if __name__ == "__main__":
    filepath = "/data/delab/georakom/"
    filename = "com-lj.ungraph.txt"
    num_workers = 4
    try:
        graph_adj, nodes, node_id_map = read_graph_to_dict(filepath + filename)
        start_time = time.time()
        total_triangles = parallel_triangle_count_master_mirror(
            graph_adj, num_workers, partition_by_master
        )
        end_time = time.time()
        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)
    except FileNotFoundError:
        print("Graph file not found.")