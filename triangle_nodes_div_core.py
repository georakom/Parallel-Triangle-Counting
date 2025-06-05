import time
from shared_mem_lib import *

# Shared-Memory Parallel Triangle Counting with TC-Merge and TC-Hash
# Work between cores is split based on num_edges / cores

def parallel_triangle_count(G, num_workers, method="merge"):
    total_start = time.time() # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    preprocess_start = time.time() # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Ranking by degree and building A+
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)
    print(f"[TIME] Preprocessing (ranking + A+): {time.time() - preprocess_start:.4f} sec") # TEST!!!!!!!!!!!!!!!!

    # Set up shared memory for CSR arrays
    shm_start = time.time() # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    shm_indptr_buf = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)
    shm_indices_buf = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)
    shm_indptr_buf[:] = indptr[:]
    shm_indices_buf[:] = indices[:]
    print(f"[TIME] Shared memory setup: {time.time() - shm_start:.4f} sec") # TEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    manager_start = time.time()
    manager = mp.Manager()
    return_dict = manager.dict()
    print(f"[TIME] Manager init: {time.time() - manager_start:.4f} sec") # TEST!!!!!!!!!!!!!!!!!!!!!!!!!
    processes = []

    # Each node's weight = length of its A⁺ set (i.e., indptr[i+1] - indptr[i])
    bucketing_start = time.time()  # HTEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if num_workers <= 32: #TESTING THAT CAN BE DELETED AFTER WARDS
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
    else:
        # Simple uniform splitting to avoid overhead
        chunk_size = len(nodes) // num_workers
        buckets = [nodes[i * chunk_size: (i + 1) * chunk_size] for i in range(num_workers)]
        remainder = len(nodes) % num_workers
        for i in range(remainder):
            buckets[i].append(nodes[-(i + 1)])  # spread leftover nodes
    print(f"[TIME] Bucketing: {time.time() - bucketing_start:.4f} sec")  # HTEST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    spawn_start = time.time()  # HTESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
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
    print(f"[TIME] Process spawning: {time.time() - spawn_start:.4f} sec")  # HTESTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    join_start = time.time()  # TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    for p in processes:
        p.join()
    print(f"[TIME] Join phase: {time.time() - join_start:.4f} sec")  # TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

    aggregation_start = time.time()  # TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    total_triangles = sum(return_dict.values())
    print(f"[TIME] Aggregating results: {time.time() - aggregation_start:.4f} sec")  # TESTTTTTTTTTTTTTTTT

    # Cleanup shared memory
    cleanup_start = time.time()  # TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    shm_indptr.close()
    shm_indptr.unlink()
    shm_indices.close()
    shm_indices.unlink()
    print(f"[TIME] Shared memory cleanup: {time.time() - cleanup_start:.4f} sec")  # TESTTTTTTTTTTTTTTTTTTTT

    print(f"[TIME] Total pipeline: {time.time() - total_start:.4f} sec")  # TESTTTTTTTTTTTTTTTTTTTTTTTTTTT
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
