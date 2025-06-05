import time
from shared_mem_lib import *

# Shared-Memory Parallel Triangle Counting with TC-Merge and TC-Hash
# Work between cores is split based on num_edges / cores

def parallel_triangle_count(G, num_workers, method="merge"):
    # Ranking by degree and building A+
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)

    # Set up shared memory for CSR arrays
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    shm_indptr_buf = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)
    shm_indices_buf = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)
    shm_indptr_buf[:] = indptr[:]
    shm_indices_buf[:] = indices[:]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    # Each node's weight = length of its A⁺ set (i.e., indptr[i+1] - indptr[i])
    weights = [(node, indptr[i + 1] - indptr[i]) for i, node in enumerate(nodes)]
    weights.sort(key=lambda x: x[1], reverse=True)  # Sort nodes by descending work

    # Initialize empty buckets for each core
    buckets = [[] for _ in range(num_workers)]
    bucket_sums = [0] * num_workers  # Keep track of total work per bucket

    for node, weight in weights:
        # Greedily assign to the bucket with the least current total weight
        min_idx = bucket_sums.index(min(bucket_sums))
        buckets[min_idx].append(node)
        bucket_sums[min_idx] += weight

    # Spawn one process per bucket
    for i in range(num_workers):
        nodes_chunk = buckets[i]
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

    # Cleanup shared memory
    shm_indptr.close()
    shm_indptr.unlink()
    shm_indices.close()
    shm_indices.unlink()

    return total_triangles

if __name__ == "__main__":
    filepath = "./data/"
    filename = "com-youtube.ungraph.txt"

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
