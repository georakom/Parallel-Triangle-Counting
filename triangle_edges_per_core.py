import time
from shared_mem_lib import *

# Shared-Memory Parallel Triangle Counting with TC-Merge and TC-Hash
# Work between cores is split based on num_edges / cores

def parallel_triangle_count(G, num_workers, method="merge"):
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)

    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)

    shm_indptr_buf = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)
    shm_indices_buf = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)

    shm_indptr_buf[:] = indptr[:]
    shm_indices_buf[:] = indices[:]

    manager = mp.Manager()
    return_dict = manager.dict()

    chunk_size = len(nodes) // num_workers
    processes = []

    for i in range(num_workers):
        start = i * chunk_size
        end = len(nodes) if i == num_workers - 1 else (i + 1) * chunk_size
        nodes_chunk = nodes[start:end]

        if method == "merge":
            p = mp.Process(target=worker_merge,
                           args=(shm_indptr.name, shm_indices.name, len(nodes),
                                 nodes_chunk, node_to_idx, return_dict))
        elif method == "hash":
            p = mp.Process(target=worker_hash,
                           args=(shm_indptr.name, shm_indices.name, len(nodes),
                                 nodes_chunk, node_to_idx, return_dict))
        else:
            raise ValueError("Method must be 'merge' or 'hash'")

        processes.append(p)
        p.start()

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
