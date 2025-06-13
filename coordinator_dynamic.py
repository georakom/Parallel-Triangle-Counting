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

        # Send results back and request next work
        task_queue.put(("triangle_count", worker_id, count))
        task_queue.put(("request_chunk", worker_id))

def coordinator(graph, num_workers, task_queue, result_queues):
    start_time = time.time()

    # Sort nodes to ensure consistent cost accumulation
    nodes_sorted = sorted(graph.nodes())
    total_cost = sum(graph.degree(n) for n in nodes_sorted)
    half_cost = total_cost // 2

    # Cost-balanced first-half split
    first_half, cumulative = [], 0
    for node in nodes_sorted:
        deg = graph.degree(node)
        if cumulative + deg > half_cost:
            break
        first_half.append(node)
        cumulative += deg

    second_half = nodes_sorted[len(first_half):]

    # Assign first-half nodes greedily to workers
    chunks = [[] for _ in range(num_workers)]
    worker_costs = [0] * num_workers
    for node in first_half:
        deg = graph.degree(node)
        i = worker_costs.index(min(worker_costs)) # Least loaded worker
        chunks[i].append(node)
        worker_costs[i] += deg

    # Send initial chunks to workers
    for i in range(num_workers):
        result_queues[i].put(("chunk", chunks[i]))
        print(f"Coordinator sent {len(chunks[i])} nodes (cost={worker_costs[i]}) to worker {i}", flush=True)

    print(f"Coordinator init finished in {time.time() - start_time:.2f} seconds")

    # This fixed cost worked for Live-Journal graph
    # # Fixed-cost chunking for second half (dynamic scheduling)
    # avg_deg = total_cost / len(nodes_sorted)
    # target_cost = max(1_000_000, min(6_000_000, int(avg_deg * 50)))  # ≈ 50 nodes worth

    # Dynamic Chunk Size Estimation Strategy
    avg_deg = total_cost / len(nodes_sorted)

    scale = len(graph.edges) / 34_681_189  # LiveJournal = baseline
    min_cost = int(1_000_000 * scale)
    max_cost = int(6_000_000 * scale)

    target_cost = max(min_cost, min(max_cost, int(avg_deg * 50)))

    vertex_chunks = []
    ptr = 0 # Pointer for nodes in second half

    while ptr < len(second_half):
        chunk, cost = [], 0
        while ptr < len(second_half) and cost < target_cost:
            node = second_half[ptr]
            chunk.append(node)
            cost += graph.degree(node)
            ptr += 1
        vertex_chunks.append(chunk)

    print(f"Coordinator pre-chunked {len(vertex_chunks)} chunks for second half.", flush=True)

    # Serve pre-made chunks on request
    chunk_ptr = 0
    pending_requests = []
    workers_done = set()
    stop_sent = set()
    triangle_counts = [0] * num_workers

    while len(workers_done) < num_workers:
        while not task_queue.empty():
            msg_type, *data = task_queue.get()
            if msg_type == "request_chunk":
                pending_requests.append(data[0])
            elif msg_type == "triangle_count":
                wid, count = data
                triangle_counts[wid] += count
            elif msg_type == "done":
                workers_done.add(data[0])

        while pending_requests and chunk_ptr < len(vertex_chunks):
            wid = pending_requests.pop(0)
            chunk = vertex_chunks[chunk_ptr]
            result_queues[wid].put(("chunk", chunk))
            print(f"Sent pre-chunked {len(chunk)} nodes to worker {wid}", flush=True)
            chunk_ptr += 1

        # Send stop if no chunks left and pending workers
        if chunk_ptr >= len(vertex_chunks):
            for wid in pending_requests:
                if wid not in stop_sent:
                    result_queues[wid].put(("stop", None))
                    stop_sent.add(wid)
            pending_requests.clear()

    print("\n=== Triangles per worker ===")
    for i, count in enumerate(triangle_counts):
        print(f"Worker {i}: {count}")
    print(f"TOTAL triangles: {sum(triangle_counts)}")



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

        num_workers = 31

        # Queues for inter-process communication
        task_queue = mp.Queue()
        result_queues = [mp.Queue() for _ in range(num_workers)]

        # Shared data (read-only dicts, passed by copy)
        shared_neighbors = {v: set(u for u in graph.adj[v] if u > v) for v in graph.nodes()}
        degrees = dict(graph.degree())

        # Launch worker processes
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

        # Begin coordinated triangle counting
        coordinator(graph, num_workers, task_queue, result_queues)

        # Join all workers
        for p in workers:
            p.join()

        print(f"In {time.time() - start:.4f} seconds")

    except FileNotFoundError:
        print("Graph file not found.")
