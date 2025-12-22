# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing as mp
import os
import statistics
import time

from tqdm import tqdm

from fastdeploy.inter_communicator.fmq import FMQ
from fastdeploy.utils import fmq_logger


# ============================================================
# Producer Task
# ============================================================
def producer_task(proc_id, msg_count, payload_size, shm_threshold, result_q):
    fmq = FMQ()
    q = fmq.queue("mp_bench_latency", role="producer")
    payload = b"x" * payload_size

    # tqdm
    pbar = tqdm(total=msg_count, desc=f"Producer-{proc_id}", position=proc_id, leave=True, disable=False)

    t0 = time.perf_counter()
    for i in range(msg_count):
        send_ts = time.perf_counter()
        q.put(data={"pid": proc_id, "i": i, "send_ts": send_ts, "payload": payload}, shm_threshold=shm_threshold)
        pbar.update(1)

    t1 = time.perf_counter()
    result_q.put({"producer_id": proc_id, "count": msg_count, "time": t1 - t0})

    pbar.close()
    time.sleep(5)
    fmq_logger.info(f"Producer-{proc_id} finished.")


# ============================================================
# Consumer Task
# ============================================================
def consumer_task(consumer_id, total_msgs, result_q, consumer_event):
    # FMQ 保持不变
    fmq = FMQ()
    q = fmq.queue("mp_bench_latency", role="consumer")
    consumer_event.set()

    latencies = []
    recv = 0

    # tqdm 显示进度
    pbar = tqdm(total=total_msgs, desc=f"Consumer-{consumer_id}", position=consumer_id + 1, leave=True, disable=False)

    first_recv = None
    last_recv = None

    while recv < total_msgs:
        try:
            msg = q.get()
        except TimeoutError:
            pbar.write("recv timeout")
            continue
        except Exception as e:
            pbar.write(f"recv failed: {e}")
            break

        recv_ts = time.perf_counter()
        if msg is None:
            pbar.write("recv None (Unexpected)")
            continue

        if first_recv is None:
            first_recv = recv_ts
        last_recv = recv_ts

        send_ts = msg["send_ts"]
        latencies.append((recv_ts - send_ts) * 1000)  # ms
        pbar.update(1)
        recv += 1

    pbar.close()

    result_q.put(
        {"consumer_id": consumer_id, "latencies": latencies, "first_recv": first_recv, "last_recv": last_recv}
    )
    fmq_logger.info(f"Consumer-{consumer_id} finished.")


# ============================================================
# MAIN benchmark
# ============================================================
def run_benchmark(
    NUM_PRODUCERS=1,
    NUM_CONSUMERS=1,
    NUM_MESSAGES_PER_PRODUCER=10000,
    PAYLOAD_SIZE=10 * 1024 * 1024,
    SHM_THRESHOLD=1 * 1024 * 1024,
):
    total_messages = NUM_PRODUCERS * NUM_MESSAGES_PER_PRODUCER
    total_bytes = total_messages * PAYLOAD_SIZE

    print(f"\nFastDeploy Message Queue Benchmark, pid:{os.getpid()}")
    print(f"Producers: {NUM_PRODUCERS}")
    print(f"Consumers: {NUM_CONSUMERS}")
    print(f"Messages per producer: {NUM_MESSAGES_PER_PRODUCER}")
    print(f"Total bytes: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    print(f"Total messages: {total_messages:,}")
    print(f"Payload per message: {PAYLOAD_SIZE / 1024 / 1024:.2f} MB")

    mp.set_start_method("fork")
    manager = mp.Manager()
    result_q = manager.Queue()

    # 两个信号事件
    consumer_event = manager.Event()

    procs = []

    # Start Consumers
    msgs_per_consumer = total_messages // NUM_CONSUMERS
    for i in range(NUM_CONSUMERS):
        p = mp.Process(target=consumer_task, args=(i, msgs_per_consumer, result_q, consumer_event))
        procs.append(p)
        p.start()

    consumer_event.wait()

    # Start Producers
    for i in range(NUM_PRODUCERS):
        p = mp.Process(
            target=producer_task, args=(i, NUM_MESSAGES_PER_PRODUCER, PAYLOAD_SIZE, SHM_THRESHOLD, result_q)
        )
        procs.append(p)
        p.start()

    # Join
    for p in procs:
        p.join()

    # Collect results
    producer_stats = []
    consumer_stats = {}

    while not result_q.empty():
        item = result_q.get()
        if "producer_id" in item:
            producer_stats.append(item)
        if "consumer_id" in item:
            consumer_stats[item["consumer_id"]] = item

    # Producer stats
    print("\nProducer Stats:")
    for p in producer_stats:
        throughput = p["count"] / p["time"]
        bandwidth = (p["count"] * PAYLOAD_SIZE) / (1024**2 * p["time"])
        print(
            f"[Producer-{p['producer_id']}] Sent {p['count']:,} msgs "
            f"in {p['time']:.3f} s | Throughput: {throughput:,.0f} msg/s | Bandwidth: {bandwidth:.2f} MB/s"
        )

    # Consumer latency stats
    print("\nConsumer Latency Stats:")
    all_latencies = []
    first_recv_times = []
    last_recv_times = []

    for cid, data in consumer_stats.items():
        lats = data["latencies"]
        if len(lats) == 0:
            continue
        all_latencies.extend(lats)
        first_recv_times.append(data["first_recv"])
        last_recv_times.append(data["last_recv"])

        try:
            p95 = statistics.quantiles(lats, n=20)[18]
            p99 = statistics.quantiles(lats, n=100)[98]
        except AttributeError:
            lats.sort()
            p95_index = int(len(lats) * 0.95)
            p99_index = int(len(lats) * 0.99)
            p95 = lats[min(p95_index, len(lats) - 1)]
            p99 = lats[min(p99_index, len(lats) - 1)]

        avg = statistics.mean(lats)
        p50 = statistics.median(lats)

        print(
            f"[Consumer-{cid}] msgs={len(lats):5d} | avg={avg:.3f} ms | "
            f"P50={p50:.3f} ms | P95={p95:.3f} ms | P99={p99:.3f} ms"
        )

    # Global summary
    if first_recv_times and last_recv_times:
        total_time = max(last_recv_times) - min(first_recv_times)
        global_throughput = total_messages / total_time
        global_bandwidth = total_bytes / (1024**2 * total_time)

        if all_latencies:
            avg_latency = statistics.mean(all_latencies)
            min_latency = min(all_latencies)
            max_latency = max(all_latencies)
            p50_latency = statistics.median(all_latencies)
            try:
                p95_latency = statistics.quantiles(all_latencies, n=20)[18]
                p99_latency = statistics.quantiles(all_latencies, n=100)[98]
            except AttributeError:
                all_latencies.sort()
                p95_index = int(len(all_latencies) * 0.95)
                p99_index = int(len(all_latencies) * 0.99)
                p95_latency = all_latencies[min(p95_index, len(all_latencies) - 1)]
                p99_latency = all_latencies[min(p99_index, len(all_latencies) - 1)]
        else:
            avg_latency = min_latency = max_latency = p50_latency = p95_latency = p99_latency = 0.0

        print("\nGlobal Summary:")
        print(f"Total messages   : {total_messages:,}")
        print(f"Total data       : {total_bytes / 1024**2:.2f} MB")
        print(f"Total time       : {total_time:.3f} s")
        print(f"Global throughput: {global_throughput:,.0f} msg/s")
        print(f"Global bandwidth : {global_bandwidth:.2f} MB/s")
        print(
            f"Latency (ms)     : avg={avg_latency:.3f} "
            f"| min={min_latency:.3f} | max={max_latency:.3f} "
            f"| P50={p50_latency:.3f} | P95={p95_latency:.3f} | P99={p99_latency:.3f}\n"
        )


# Entry
if __name__ == "__main__":
    run_benchmark()
