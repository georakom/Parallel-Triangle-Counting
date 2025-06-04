import time
from shared_mem_lib import *

# Shared-Memory Parallel Triangle Counting is implemented with TC-Merge and TC-Hash
# Work between cores is split based on a specific number (for example 5000 edges) for a better balance

def parallel_triangle_count(G, num_workers, method="merge", chunk_size=50000):
    rank = rank_by_degree(G)
    indptr, indices, nodes, node_to_idx = build_A_plus_csr(G, rank)

    # Shared memory setup
    shm_indptr = shared_memory.SharedMemory(create=True, size=indptr.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=indices.nbytes)
    np.ndarray(indptr.shape, dtype=indptr.dtype, buffer=shm_indptr.buf)[:] = indptr[:]
    np.ndarray(indices.shape, dtype=indices.dtype, buffer=shm_indices.buf)[:] = indices[:]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    sem = mp.Semaphore(num_workers)

    # Schedule smaller dynamic chunks like OpenMP dynamic schedule
    for i in range(0, len(nodes), chunk_size):
        nodes_chunk = nodes[i:i + chunk_size]
        sem.acquire()
        p = mp.Process(target=worker_wrapper, args=(
            method,
            shm_indptr.name,
            shm_indices.name,
            len(nodes),
            nodes_chunk,
            node_to_idx,
            return_dict,
            sem
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_triangles = sum(return_dict.values())

    shm_indptr.close()
    shm_indptr.unlink()
    shm_indices.close()
    shm_indices.unlink()

    return total_triangles

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
