import time
import numpy as np
from numba import njit

def read_graph_edges(filename):
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
    return remapped_edges, nodes, node_id_map

def get_ordering(num_nodes, remapped_edges):
    # Build degree array
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for u, v in remapped_edges:
        degrees[u] += 1
        degrees[v] += 1
    order = np.lexsort((np.arange(num_nodes), degrees))
    node_to_order = {node: rank for rank, node in enumerate(order)}
    order_to_node = {rank: node for rank, node in enumerate(order)}
    return node_to_order, order_to_node

def partition_by_master(num_nodes, node_to_order, num_workers):
    # Assign master by order
    order_to_host = {rank: rank % num_workers for rank in range(num_nodes)}
    partitions = [[] for _ in range(num_workers)]
    assignments = np.zeros(num_nodes, dtype=int)
    for node in range(num_nodes):
        host = order_to_host[node_to_order[node]]
        partitions[host].append(node)
        assignments[node] = host
    return partitions, assignments

def build_worker_adjacency_lists(num_nodes, remapped_edges, partitions, node_to_order):
    # First, build full adjacency for all nodes (undirected)
    adj = [[] for _ in range(num_nodes)]
    for u, v in remapped_edges:
        adj[u].append(v)
        adj[v].append(u)
    # For each worker, build {local_id: neighbors (sorted)} for needed nodes (masters + mirrors)
    worker_adj_lists = []
    worker_master_nodes = []
    worker_global_to_local = []
    worker_local_to_global = []
    for master_nodes in partitions:
        nodes_needed = set(master_nodes)
        for u in master_nodes:
            nodes_needed.update(adj[u])  # add all neighbors (mirrors)
        nodes_needed = sorted(nodes_needed)
        global_to_local = {global_id: local_id for local_id, global_id in enumerate(nodes_needed)}
        local_to_global = {local_id: global_id for global_id, local_id in global_to_local.items()}
        # Build local adjacency lists (as lists of local_ids, sorted)
        local_adj = []
        for global_id in nodes_needed:
            neighbors = [global_to_local[v] for v in adj[global_id] if v in global_to_local]
            local_adj.append(np.array(sorted(neighbors), dtype=np.int32))
        master_local_ids = {global_to_local[u] for u in master_nodes}
        worker_adj_lists.append(local_adj)
        worker_master_nodes.append(master_local_ids)
        worker_global_to_local.append(global_to_local)
        worker_local_to_global.append(local_to_global)
    return worker_adj_lists, worker_master_nodes, worker_global_to_local, worker_local_to_global

def adjlists_to_csr(adjlists):
    n = len(adjlists)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indices = []
    for i, nbrs in enumerate(adjlists):
        indptr[i+1] = indptr[i] + len(nbrs)
        indices.extend(nbrs)
    indices = np.array(indices, dtype=np.int32)
    return indptr, indices

@njit
def count_triangles_master_only(indptr, indices, master_mask, order_array, local_to_global_array):
    count = 0
    n = len(local_to_global_array)
    for u_local in range(n):
        if not master_mask[u_local]:
            continue
        u_global = local_to_global_array[u_local]
        u_order = order_array[u_global]
        u_neighbors = indices[indptr[u_local]:indptr[u_local+1]]
        for v_local in u_neighbors:
            v_global = local_to_global_array[v_local]
            v_order = order_array[v_global]
            if v_order <= u_order:
                continue
            v_neighbors = indices[indptr[v_local]:indptr[v_local+1]]
            # Only consider w > v (total order)
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
    indptr, indices, master_mask, order_array, local_to_global_array = args
    return count_triangles_master_only(indptr, indices, master_mask, order_array, local_to_global_array)

def parallel_triangle_count_master_mirror(filename, num_workers):
    t0 = time.time()
    remapped_edges, nodes, node_id_map = read_graph_edges(filename)
    num_nodes = len(nodes)
    t1 = time.time()
    node_to_order, order_to_node = get_ordering(num_nodes, remapped_edges)
    t2 = time.time()
    partitions, assignments = partition_by_master(num_nodes, node_to_order, num_workers)
    t3 = time.time()
    worker_adj_lists, worker_master_nodes, worker_global_to_local, worker_local_to_global = build_worker_adjacency_lists(
        num_nodes, remapped_edges, partitions, node_to_order)
    t4 = time.time()
    # Prepare data for each worker (adjlist -> CSR arrays)
    worker_args = []
    N = num_nodes
    order_array = np.zeros(N, dtype=np.int32)
    for node, order in node_to_order.items():
        order_array[node] = order
    for wid in range(num_workers):
        adjlists = worker_adj_lists[wid]
        indptr, indices = adjlists_to_csr(adjlists)
        n = len(adjlists)
        master_mask = np.zeros(n, dtype=np.bool_)
        for idx in worker_master_nodes[wid]:
            master_mask[idx] = True
        # local_to_global_array: local id -> compact id
        local_to_global = worker_local_to_global[wid]
        local_to_global_array = np.zeros(n, dtype=np.int32)
        for local, global_id in local_to_global.items():
            local_to_global_array[local] = global_id
        worker_args.append((indptr, indices, master_mask, order_array, local_to_global_array))
    t5 = time.time()
    print(f"Edge reading: {t1-t0:.2f}s, Ordering: {t2-t1:.2f}s, Partitioning: {t3-t2:.2f}s, Adjacency build: {t4-t3:.2f}s, Final prep: {t5-t4:.2f}s")
    # Triangle counting in parallel
    from multiprocessing import Pool, get_context
    triangle_count_time = time.time()
    with get_context("fork").Pool(num_workers) as pool:
        results = pool.map(count_triangles_worker, worker_args)
    print(f"Triangle counting took {time.time() - triangle_count_time:.2f} seconds")
    for wid, count in enumerate(results):
        print(f"Worker {wid} counted {count} triangles")
    return sum(results)

if __name__ == "__main__":
    filepath = "/data/delab/georakom"
    filename = "com-lj.ungraph.txt"
    num_workers = 4
    try:
        start_time = time.time()
        total_triangles = parallel_triangle_count_master_mirror(filepath + filename, num_workers)
        end_time = time.time()
        print(f"Total triangles: {total_triangles}")
        print("Triangle Algorithm time: ", end_time - start_time)
    except FileNotFoundError:
        print("Graph file not found.")