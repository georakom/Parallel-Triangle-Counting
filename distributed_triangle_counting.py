import random
import time
import networkx as nx
import numpy as np

from numba import njit, types
from numba.typed import List, Dict

@njit
def build_worker_edges(indptr, indices, assignments_array, partitions_list, num_workers):
    worker_edges = List()
    worker_nodes = List()
    for _ in range(num_workers):
        worker_edges.append(List.empty_list(types.UniTuple(types.int32, 2)))
        worker_nodes.append(set())

    for u in range(len(indptr) - 1):
        master_u = assignments_array[u]
        start_u, end_u = indptr[u], indptr[u + 1]
        neighbors_u = indices[start_u:end_u]

        # Assign (u, v)
        for i in range(start_u, end_u):
            v = indices[i]
            if u < v:
                worker_edges[master_u].append((u, v))
                worker_nodes[master_u].add(u)
                worker_nodes[master_u].add(v)

        # Triangle closure via (v, w)
        for i in range(start_u, end_u):
            v = indices[i]
            if v <= u:
                continue
            start_v, end_v = indptr[v], indptr[v + 1]
            neighbors_v = indices[start_v:end_v]

            for j in range(i + 1, end_u):
                w = indices[j]
                if w <= v:
                    continue
                # in-place merge (v, w)
                vi = 0
                while vi < len(neighbors_v):
                    if neighbors_v[vi] == w:
                        worker_edges[master_u].append((v, w))
                        worker_nodes[master_u].add(v)
                        worker_nodes[master_u].add(w)
                        break
                    elif neighbors_v[vi] < w:
                        vi += 1
                    else:
                        break

    return worker_edges, worker_nodes

def convert_nx_to_csr(G):
    node_list = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n = len(node_list)

    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        uid, vid = node_to_idx[u], node_to_idx[v]
        adj[uid].append(vid)
        adj[vid].append(uid)

    for neighbors in adj:
        neighbors.sort()

    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = []

    for i, neighbors in enumerate(adj):
        indptr[i + 1] = indptr[i] + len(neighbors)
        indices.extend(neighbors)

    indices = np.array(indices, dtype=np.int32)
    return indptr, indices, node_to_idx, idx_to_node

def extract_all_worker_data_csr(indptr, indices, partitions, assignments, num_workers):
    import numpy as np
    from numba.typed import List

    num_nodes = len(indptr) - 1

    # Convert dict to array for numba
    assignments_array = np.empty(num_nodes, dtype=np.int32)
    for u in range(num_nodes):
        assignments_array[u] = assignments[u]

    # Convert partitions to set for mirror detection
    partition_sets = [set(p) for p in partitions]

    # Call njit core
    worker_edges, worker_nodes = build_worker_edges(indptr, indices, assignments_array, partitions, num_workers)

    # Convert to Python format
    worker_data = []
    for wid in range(num_workers):
        masters = partition_sets[wid]
        mirrors = worker_nodes[wid] - masters
        print(f"[Worker {wid}] Mirror nodes: {len(mirrors)}")
        worker_data.append((list(worker_edges[wid]), masters))

    return worker_data


def count_worker_triangles(indptr, indices, master_nodes):
    from numba import njit
    import numpy as np

    @njit
    def count(indptr, indices, masters):
        count = 0
        for u in masters:
            neighbors_u = indices[indptr[u]:indptr[u+1]]
            for i in range(len(neighbors_u)):
                v = neighbors_u[i]
                if v <= u:
                    continue
                neighbors_v = indices[indptr[v]:indptr[v+1]]
                j = i + 1
                for j in range(i + 1, len(neighbors_u)):
                    w = neighbors_u[j]
                    if w <= v:
                        continue
                    # check if w in neighbors_v via merge
                    vi = 0
                    while vi < len(neighbors_v):
                        if neighbors_v[vi] == w:
                            count += 1
                            break
                        elif neighbors_v[vi] < w:
                            vi += 1
                        else:
                            break
        return count

    return count(indptr, indices, np.array(list(master_nodes), dtype=np.int32))



def parallel_triangle_count(G, num_workers, partition_func):
    import gc
    import numpy as np

    start = time.time()
    partitions, assignments = partition_func(G, num_workers)
    indptr, indices, node_to_idx, idx_to_node = convert_nx_to_csr(G)
    del G
    gc.collect()

    # Remap partitions
    partitions = [[node_to_idx[n] for n in part] for part in partitions]

    from multiprocessing import Pool
    triangle_time = time.time()
    args = [(indptr, indices, set(part)) for part in partitions]
    with Pool(num_workers) as pool:
        results = pool.starmap(count_worker_triangles, args)
    print(f"Triangle counting took: {time.time() - triangle_time:.4f} seconds")
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