from shared_mem_lib import *
import heapq
import multiprocessing as mp

# Shared-Memory Parallel Triangle Counting with TC-Merge and TC-Hash
# Work between cores is split based on A+ weight

def parallel_triangle_count(G, num_workers, method="merge"):
    total_start = time.time()
    preprocess_start = time.time()
    # Ranking by degree and building A+
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)
    print(f"[TIME] Preprocessing (ranking + A+): {time.time() - preprocess_start:.4f} sec")

    # Set up shared memory for CSR arrays
    shm_start = time.time()
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    shm_indptr_buf = np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)
    shm_indices_buf = np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)
    shm_indptr_buf[:] = indptr[:]
    shm_indices_buf[:] = indices[:]
    print(f"[TIME] Shared memory setup: {time.time() - shm_start:.4f} sec")

    manager_start = time.time() # QUEUE TEST
    # manager = mp.Manager()
    # return_dict = manager.dict()
    queue = mp.Queue()
    print(f"[TIME] Manager init: {time.time() - manager_start:.4f} sec")
    processes = []

    bucketing_start = time.time()

    # Estimate work for each node
    weights = [(node, indptr[i + 1] - indptr[i]) for i, node in enumerate(nodes)]

    # Sort by heaviest first (LPT first based)
    weights.sort(key=lambda x: x[1], reverse=True)  # High-work nodes first

    # Min-heap (workload_sum, core_index), stores current load per core
    buckets = [[] for _ in range(num_workers)]
    heap = [(0, i) for i in range(num_workers)]  # (current_work_sum, core_id)
    heapq.heapify(heap)

    # Assign each node to the least loaded core
    for node, weight in weights:
        curr_sum, core_id = heapq.heappop(heap)
        buckets[core_id].append(node)
        heapq.heappush(heap, (curr_sum + weight, core_id))
    print(f"[TIME] Bucketing: {time.time() - bucketing_start:.4f} sec")

    spawn_start = time.time()

    for i in range(num_workers):
        nodes_chunk = buckets[i]
        if method == "merge": # QUEUE TEST
            p = mp.Process(target=worker_merge,
                            args=(shm_indptr.name, shm_indices.name, len(nodes),
                                    nodes_chunk, node_to_idx, queue))
        elif method == "hash": # QUEUE TEST
            p = mp.Process(target=worker_hash,
                            args=(shm_indptr.name, shm_indices.name, len(nodes),
                                    nodes_chunk, node_to_idx, queue))
        else:
            raise ValueError("Method must be 'merge' or 'hash'")
        processes.append(p)
        p.start()
    print(f"[TIME] Process spawning: {time.time() - spawn_start:.4f} sec")

    join_start = time.time()
    for p in processes:
        p.join()
    print(f"[TIME] Join phase: {time.time() - join_start:.4f} sec")

    aggregation_start = time.time() # QUEUE TEST
    # total_triangles = sum(return_dict.values())
    total_triangles = 0
    for _ in range(num_workers):
        total_triangles += queue.get()
    print(f"[TIME] Aggregating results: {time.time() - aggregation_start:.4f} sec")

    # Cleanup shared memory
    cleanup_start = time.time()
    shm_indptr.close()
    shm_indptr.unlink()
    shm_indices.close()
    shm_indices.unlink()
    print(f"[TIME] Shared memory cleanup: {time.time() - cleanup_start:.4f} sec")

    print(f"[TIME] Total pipeline: {time.time() - total_start:.4f} sec")
    return total_triangles

if __name__ == "__main__":
    filepath = "/data/delab/georakom/"
    filename = "lfriendster.txt"

    try:
        graph = read_graph_from_file(filepath + filename)

        for method in ["merge", "hash"]:
            print(f"\n===== Running TC-{method.upper()} =====")
            start_time = time.time()
            total_triangles = parallel_triangle_count(graph, 4, method=method)
            end_time = time.time()
            print(f"Total triangles ({method}): {total_triangles}")
            print(f"Triangle Algorithm time ({method}): {end_time - start_time:.4f} seconds")

        # print("\n===== NetworkX Verification =====")
        # start_time = time.time()
        # triangles_networkx = nx.triangles(graph)
        # triangles_networkx_count = sum(triangles_networkx.values()) // 3
        # end_time = time.time()
        # print(f"Total triangles (NetworkX): {triangles_networkx_count}")
        # print(f"Triangle Algorithm time (NetworkX): {end_time - start_time:.4f} seconds")

    except FileNotFoundError:
        print("File not found")
