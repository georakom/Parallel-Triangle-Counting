import networkx as nx
import multiprocessing as mp
import time
import random
import queue

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


def coordinator(graph, num_workers, task_queue, result_queues):
    init_time = time.time()

    # ---- Initial Preparation ----
    # Deterministically sort nodes
    nodes_sorted = sorted(graph.nodes())

    total_cost = compute_total_cost(nodes_sorted, graph)
    half_cost = total_cost // 2

    first_half = []
    cost_so_far = 0
    for node in nodes_sorted:
        node_cost = graph.degree(node)
        if cost_so_far + node_cost <= half_cost:
            first_half.append(node)
            cost_so_far += node_cost
        else:
            break

    second_half = nodes_sorted[len(first_half):]

    # ---- Assign Initial Chunks (First Half) ----
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

    # ---- Coordinator Runtime State ----
    ptr = 0  # Pointer into second_half
    k = 1.0
    worker_costs = costs.copy()
    workers_sent_stop = set()
    workers_done = set()
    triangle_counts = [0] * num_workers
    pending_requests = []

    MIN_CHUNK_NODES = 50000
    MIN_CHUNK_COST = 2000000
    ready_chunks = queue.Queue(maxsize=2 * num_workers)  # Pre-buffered chunks

    def prepare_next_chunk():
        """Prepare a chunk from the second_half and enqueue it for later dispatch."""
        nonlocal ptr, k

        if ptr >= len(second_half):
            return False  # All work done

        remaining_cost = compute_total_cost(second_half[ptr:], graph)
        target_cost = max(MIN_CHUNK_COST, int(remaining_cost / (num_workers * k)))
        k *= 1.2

        chunk, cost = [], 0
        while ptr < len(second_half) and cost < target_cost:
            node = second_half[ptr]
            chunk.append(node)
            cost += graph.degree(node)
            ptr += 1

        # If close to end, flush remaining
        if len(second_half) - ptr < MIN_CHUNK_NODES:
            while ptr < len(second_half):
                node = second_half[ptr]
                chunk.append(node)
                cost += graph.degree(node)
                ptr += 1

        ready_chunks.put((chunk, cost))
        return True

    # ---- Main Loop ----
    while len(workers_done) < num_workers:
        # --- Respond to worker messages ---
        while not task_queue.empty():
            msg_type, *data = task_queue.get()
            if msg_type == "request_chunk":
                pending_requests.append(data[0])
            elif msg_type == "triangle_count":
                worker_id, count = data
                triangle_counts[worker_id] += count
            elif msg_type == "done":
                worker_id = data[0]
                workers_done.add(worker_id)

        # --- Dispatch precomputed chunks ---
        while pending_requests and not ready_chunks.empty():
            worker_id = pending_requests.pop(0)
            chunk, cost = ready_chunks.get()
            result_queues[worker_id].put(("chunk", chunk))
            worker_costs[worker_id] += cost
            print(f"Coordinator assigned {len(chunk)} nodes (cost={cost}) to worker {worker_id}", flush=True)

        # --- Prepare new chunks if buffer not full ---
        if ptr < len(second_half) and not ready_chunks.full():
            prepare_next_chunk()

        # --- If no work remains, send "stop" to all pending workers ---
        if ptr >= len(second_half) and ready_chunks.empty() and pending_requests:
            for worker_id in pending_requests:
                if worker_id not in workers_sent_stop:
                    result_queues[worker_id].put(("stop", None))
                    workers_sent_stop.add(worker_id)
            pending_requests.clear()

        # Sleep briefly to yield control and reduce busy waiting
        time.sleep(0.001)

    # ---- Final Reporting ----
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
    filepath = "/data/delab/georakom/"
    filename = "com-lj.ungraph.txt"

    try:
        graph = read_graph_from_file(filepath + filename )

        num_workers = 32

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
