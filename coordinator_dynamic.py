import networkx as nx
import multiprocessing as mp
import time
import random

def compute_local_triangles(shared_neighbors, nodes):
    triangle_count = 0
    for u in nodes:
        neighbors_u = shared_neighbors.get(u, set())
        for v in neighbors_u:
            neighbors_v = shared_neighbors.get(v, set())
            triangle_count += len(neighbors_u & neighbors_v)
    return triangle_count


def worker(worker_id, shared_neighbors, task_queue, result_queue):
    while True:
        msg_type, payload = result_queue.get()
        if msg_type == "stop":
            task_queue.put(("done", worker_id))
            break
        elif msg_type != "chunk":
            raise RuntimeError("Unexpected message")

        tri_time = time.time()
        chunk = payload
        count = compute_local_triangles(shared_neighbors, chunk)
        print(f"Worker {worker_id} processed {len(chunk)} nodes, found {count} triangles in {time.time() - tri_time:.4f} seconds", flush=True)


        task_queue.put(("triangle_count", worker_id, count))
        task_queue.put(("request_chunk", worker_id))


def compute_total_cost(nodes, G):
    return sum(G.degree(n) for n in nodes)


# def coordinator(graph, num_workers, task_queue, result_queues):
#     init_time = time.time()
#
#     # Sort nodes for deterministic partitioning
#     nodes_sorted = sorted(graph.nodes())
#
#     # Split work into two phases by cost (degree sum)
#     total_cost = compute_total_cost(graph.nodes(), graph)
#     half_cost = total_cost // 2
#
#     first_half = []
#     first_half_cost = 0
#     for node in nodes_sorted:
#         cost = graph.degree(node)
#         if first_half_cost + cost <= half_cost:
#             first_half.append(node)
#             first_half_cost += cost
#         else:
#             break
#
#     second_half_start_idx = len(first_half)
#     second_half = list(nodes_sorted)[second_half_start_idx:]
#
#     # INITIAL WORK: Evenly split first half
#     chunks = [[] for _ in range(num_workers)]
#     costs = [0] * num_workers
#     for node in first_half:
#         min_worker = costs.index(min(costs))
#         chunks[min_worker].append(node)
#         costs[min_worker] += graph.degree(node)
#
#     for i in range(num_workers):
#         result_queues[i].put(("chunk", chunks[i]))
#         print(f"Coordinator initially assigned {len(chunks[i])} nodes (cost={compute_total_cost(chunks[i], graph)}) to worker {i}", flush=True)
#     print(f"Initialization time: {time.time() - init_time:.4f} seconds")
#
#     ptr = 0
#     k = 1.0
#     worker_costs = costs.copy()  # Already did first-half work
#     workers_sent_stop = set()
#     workers_reported_done = set()
#     triangle_counts = [0] * num_workers
#     MIN_CHUNK_NODES = 20000
#     MIN_CHUNK_COST = 20000
#     pending_requests = []
#
#     while len(workers_reported_done) < num_workers:
#         while not task_queue.empty():
#             msg_type, *data = task_queue.get()
#             if msg_type == "request_chunk":
#                 pending_requests.append(data[0])
#             elif msg_type == "triangle_count":
#                 worker_id, count = data
#                 triangle_counts[worker_id] += count
#             elif msg_type == "done":
#                 worker_id = data[0]
#                 workers_reported_done.add(worker_id)
#
#         if pending_requests and ptr < len(second_half):
#             # Choose the least-loaded worker from pending
#             best_worker = min(pending_requests, key=lambda wid: worker_costs[wid])
#             pending_requests.remove(best_worker)
#
#             remaining_cost = compute_total_cost(second_half[ptr:], graph)
#             # Comments to test
#             target_cost = max(MIN_CHUNK_COST, int(remaining_cost / (num_workers * k)))
#             k *= 1.5
#
#             chunk, cost = [], 0
#             while ptr < len(second_half) and cost < target_cost:
#                 node = second_half[ptr]
#                 chunk.append(node)
#                 cost += graph.degree(node)
#                 ptr += 1
#
#             if len(second_half) - ptr < MIN_CHUNK_NODES:
#                 while ptr < len(second_half):
#                     node = second_half[ptr]
#                     chunk.append(node)
#                     cost += graph.degree(node)
#                     ptr += 1
#
#             result_queues[best_worker].put(("chunk", chunk))
#             worker_costs[best_worker] += cost
#             print(f"Coordinator assigned {len(chunk)} nodes (cost={cost}) to worker {best_worker}", flush=True)
#
#         # If no work remains, send "stop" to anyone still requesting
#         if ptr >= len(second_half) and pending_requests:
#             for wid in pending_requests:
#                 if wid not in workers_sent_stop:
#                     result_queues[wid].put(("stop", None))
#                     workers_sent_stop.add(wid)
#             pending_requests.clear()
#
#     print("\nTriangles per worker:")
#     for i, count in enumerate(triangle_counts):
#         print(f"  Worker {i}: {count}")
#     total = sum(triangle_counts)
#     print(f"\nTotal triangles: {total}")

def coordinator(graph, num_workers, task_queue, result_queues):
    import time
    init_time = time.time()

    # Sort nodes for deterministic partitioning
    nodes_sorted = sorted(graph.nodes())

    # Split work into two halves by total degree cost
    total_cost = compute_total_cost(graph.nodes(), graph)
    half_cost = total_cost // 2

    first_half = []
    first_half_cost = 0
    for node in nodes_sorted:
        cost = graph.degree(node)
        if first_half_cost + cost <= half_cost:
            first_half.append(node)
            first_half_cost += cost
        else:
            break

    second_half_start_idx = len(first_half)
    second_half = list(nodes_sorted)[second_half_start_idx:]

    # Initial work: distribute first half based on degree cost
    chunks = [[] for _ in range(num_workers)]
    costs = [0] * num_workers
    for node in first_half:
        min_worker = costs.index(min(costs))
        chunks[min_worker].append(node)
        costs[min_worker] += graph.degree(node)

    for i in range(num_workers):
        result_queues[i].put(("chunk", chunks[i]))
        print(f"Coordinator initially assigned {len(chunks[i])} nodes (cost={compute_total_cost(chunks[i], graph)}) to worker {i}", flush=True)
    print(f"Initialization time: {time.time() - init_time:.4f} seconds")

    # Begin dynamic scheduling of second half
    ptr = 0
    k = 1.1  # Shrinking factor
    worker_costs = costs.copy()
    workers_sent_stop = set()
    workers_reported_done = set()
    triangle_counts = [0] * num_workers

    MIN_CHUNK_NODES = 20000
    MIN_CHUNK_COST = 20000

    pending_requests = []
    prebuilt_chunks = []

    while len(workers_reported_done) < num_workers:
        # Handle incoming messages
        while not task_queue.empty():
            msg_type, *data = task_queue.get()
            if msg_type == "request_chunk":
                pending_requests.append(data[0])
            elif msg_type == "triangle_count":
                worker_id, count = data
                triangle_counts[worker_id] += count
            elif msg_type == "done":
                worker_id = data[0]
                workers_reported_done.add(worker_id)

        # Prebuild next chunk (up to buffer limit)
        if ptr < len(second_half) and len(prebuilt_chunks) < 2:
            remaining_cost = compute_total_cost(second_half[ptr:], graph)
            effective_workers = max(1, num_workers - 1)
            target_cost = max(MIN_CHUNK_COST, int(remaining_cost / (effective_workers * k)))
            k *= 1.2  # Increase divisor to shrink chunks

            chunk, cost = [], 0
            while ptr < len(second_half) and cost < target_cost:
                node = second_half[ptr]
                chunk.append(node)
                cost += graph.degree(node)
                ptr += 1

            if len(second_half) - ptr < MIN_CHUNK_NODES:
                while ptr < len(second_half):
                    node = second_half[ptr]
                    chunk.append(node)
                    cost += graph.degree(node)
                    ptr += 1

            prebuilt_chunks.append((chunk, cost))

        # Serve pending chunk requests from prebuilt chunks
        while pending_requests and prebuilt_chunks:
            best_worker = min(pending_requests, key=lambda wid: worker_costs[wid])
            pending_requests.remove(best_worker)

            chunk, cost = prebuilt_chunks.pop(0)
            result_queues[best_worker].put(("chunk", chunk))
            worker_costs[best_worker] += cost
            print(f"Coordinator assigned {len(chunk)} nodes (cost={cost}) to worker {best_worker}", flush=True)

        # If no work remains, send "stop" to any remaining pending workers
        if ptr >= len(second_half) and not prebuilt_chunks and pending_requests:
            for wid in pending_requests:
                if wid not in workers_sent_stop:
                    result_queues[wid].put(("stop", None))
                    workers_sent_stop.add(wid)
            pending_requests.clear()

    print("\nTriangles per worker:")
    for i, count in enumerate(triangle_counts):
        print(f"  Worker {i}: {count}")
    total = sum(triangle_counts)
    print(f"\nTotal triangles: {total}")


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
        graph = read_graph_from_file(filepath + filename )

        num_workers = 3

        # Use mp.Queue (faster) and no manager at all
        task_queue = mp.Queue()
        result_queues = [mp.Queue() for _ in range(num_workers)]

        # Shared data (read-only dicts, passed by copy)
        shared_neighbors = {v: set(u for u in graph.adj[v] if u > v) for v in graph.nodes()}
        degrees = dict(graph.degree())

        workers = []
        start = time.time()

        for worker_id in range(num_workers):
            p = mp.Process(target=worker, args=(
                worker_id, shared_neighbors,
                task_queue, result_queues[worker_id]
            ))
            p.start()
            workers.append(p)

        print(f"Worker spawning took: {time.time() - start:.4f} seconds.")

        coordinator(graph, num_workers, task_queue, result_queues)

        for p in workers:
            p.join()

        print(f"In {time.time() - start:.4f} seconds")

    except FileNotFoundError:
        print("Graph file not found.")
