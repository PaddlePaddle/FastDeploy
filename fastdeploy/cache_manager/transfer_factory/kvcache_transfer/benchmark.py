"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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
"""

"""
KV Cache Transfer Benchmark:
Performance test for measuring KV cache transfer throughput (GB/s)

Usage:
    # Start decode server first:
    python benchmark.py --splitwise_role decode --decode_rdma_port 9881 --decode_zmq_port 9882

    # Then start prefill client:
    python benchmark.py --splitwise_role prefill --decode_ip <decode_ip> --decode_rdma_port 9881 --decode_zmq_port 9882
"""

import argparse
import time
from dataclasses import dataclass
from typing import List

import paddle
import rdma_comm
import zmq

if paddle.is_compiled_with_xpu():
    from custom_setup_ops import get_peer_mem_addr


@dataclass
class BenchmarkConfig:
    """Benchmark configuration parameters"""

    num_layers: int = 61  # Number of transformer layers
    max_block_num: int = 2000  # Maximum number of blocks
    kv_num_head: int = 1  # Number of KV heads
    block_size_seq_len: int = 64  # Block size (sequence length dimension)
    head_dim: int = 128  # Hidden size per head
    cache_dtype: str = "bfloat16"  # Cache data type: bfloat16, float16, uint8
    warmup_iters: int = 5  # Warmup iterations
    benchmark_iters: int = 20  # Benchmark iterations
    blocks_per_transfer: int = 50  # Number of blocks per transfer
    num_queries: int = 1  # Number of queries per layer (simulate multiple queries)


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    total_bytes: int
    elapsed_time: float
    throughput_gbps: float
    iterations: int
    blocks_per_iter: int
    layers: int
    queries_per_layer: int


class KVCacheBenchmark:
    def __init__(self, splitwise_role: str, config: BenchmarkConfig, port: int = None, device: str = "gpu"):
        assert splitwise_role in ["prefill", "decode"], "splitwise_role must be prefill or decode"
        if splitwise_role == "decode":
            assert port, "port must be specified for decode server"

        self.splitwise_role = splitwise_role
        self.config = config
        self.gpu_cache_kvs = {}
        paddle.device.set_device(device)

        print(f"[Benchmark] Role: {splitwise_role}, Port: {port}, Device: {device}")
        print(
            f"[Benchmark] Config: layers={config.num_layers}, max_blocks={config.max_block_num}, "
            f"kv_heads={config.kv_num_head}, block_size={config.block_size_seq_len}, "
            f"head_dim={config.head_dim}, dtype={config.cache_dtype}"
        )

        # Determine cache dtype
        if paddle.is_compiled_with_xpu():
            cache_type = "float16"
        else:
            cache_type = config.cache_dtype

        cache_k_ptr_list = []
        cache_v_ptr_list = []
        cache_k_scale_ptr_list = []
        cache_v_scale_ptr_list = []

        # Calculate block size
        block_shape = [
            config.max_block_num,
            config.kv_num_head,
            config.block_size_seq_len,
            config.head_dim,
        ]
        scale_shape = [
            config.max_block_num,
            config.kv_num_head,
            config.block_size_seq_len,
        ]

        for layer_idx in range(config.num_layers):
            # Create key cache
            key_cache = paddle.zeros(shape=block_shape, dtype=cache_type)
            self.gpu_cache_kvs[f"key_caches_{layer_idx}"] = key_cache
            if paddle.is_compiled_with_xpu():
                cache_k_ptr_list.append(get_peer_mem_addr(key_cache.data_ptr()))
            else:
                cache_k_ptr_list.append(key_cache.data_ptr())

            # Create value cache
            value_cache = paddle.zeros(shape=block_shape, dtype=cache_type)
            self.gpu_cache_kvs[f"value_caches_{layer_idx}"] = value_cache
            if paddle.is_compiled_with_xpu():
                cache_v_ptr_list.append(get_peer_mem_addr(value_cache.data_ptr()))
            else:
                cache_v_ptr_list.append(value_cache.data_ptr())

            # Create scale tensors
            key_scale = paddle.zeros(shape=scale_shape, dtype="float32")
            self.gpu_cache_kvs[f"key_scale_{layer_idx}"] = key_scale
            if paddle.is_compiled_with_xpu():
                cache_k_scale_ptr_list.append(get_peer_mem_addr(key_scale.data_ptr()))
            else:
                cache_k_scale_ptr_list.append(key_scale.data_ptr())

            value_scale = paddle.zeros(shape=scale_shape, dtype="float32")
            self.gpu_cache_kvs[f"value_scale_{layer_idx}"] = value_scale
            if paddle.is_compiled_with_xpu():
                cache_v_scale_ptr_list.append(get_peer_mem_addr(value_scale.data_ptr()))
            else:
                cache_v_scale_ptr_list.append(value_scale.data_ptr())

            # Initialize prefill data with pattern for validation
            if self.splitwise_role == "prefill":
                key_cache.fill_(1.0)
                value_cache.fill_(1.0)

        # Calculate block bytes
        dtype_bytes = 2 if cache_type in ["bfloat16", "float16"] else 1
        block_bytes = config.kv_num_head * config.block_size_seq_len * config.head_dim * dtype_bytes
        scale_block_bytes = config.kv_num_head * config.block_size_seq_len * 4  # float32

        self.block_bytes = block_bytes
        self.scale_block_bytes = scale_block_bytes
        self.bytes_per_block = block_bytes * 2 + scale_block_bytes * 2  # K + V + K_scale + V_scale

        print(f"[Benchmark] Block bytes: {block_bytes}, Scale bytes: {scale_block_bytes}")
        print(f"[Benchmark] Total bytes per block (K+V+scales): {self.bytes_per_block}")
        print(
            f"[Benchmark] Total cache memory: {self._format_bytes(self.bytes_per_block * config.max_block_num * config.num_layers)}"
        )

        # Create RDMA communicator
        self.rdma_comm = rdma_comm.RDMACommunicator(
            splitwise_role,
            0,
            str(port) if self.splitwise_role == "decode" else "0",
            cache_k_ptr_list,
            cache_v_ptr_list,
            config.max_block_num,
            block_bytes,
            cache_k_scale_ptr_list,
            cache_v_scale_ptr_list,
            scale_block_bytes,
        )

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes to human readable string"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"

    def is_connected(self, ip: str, port: int) -> bool:
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        return self.rdma_comm.is_connected(ip, str(port))

    def connect(self, ip: str, port: int) -> bool:
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        if self.is_connected(ip, port):
            return True
        self.rdma_comm.connect(ip, str(port))
        return True

    def write_cache(self, ip: str, port: int, query_block_ids_list: List[List[int]]):
        """Write cache to remote with all layers and queries

        Args:
            ip: Remote IP address
            port: Remote port
            query_block_ids_list: List of block_ids for each query, length should be num_queries
        """
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        assert (
            len(query_block_ids_list) == self.config.num_queries
        ), f"Expected {self.config.num_queries} query block_ids, got {len(query_block_ids_list)}"

        for layer_idx in range(self.config.num_layers):
            for query_idx in range(self.config.num_queries):
                block_ids = query_block_ids_list[query_idx]
                self.rdma_comm.write_cache(ip, str(port), block_ids, block_ids, layer_idx)

    def _generate_query_block_ids(self, blocks_per_query: int) -> List[List[int]]:
        """Generate block_ids for each query with gaps between queries

        Each query has contiguous block_ids, but different queries are separated by gaps.
        For example, with 2 queries, 3 blocks each, and gap=10:
            query 0: [0, 1, 2]
            query 1: [13, 14, 15]  (starts at 0 + 3 + 10 = 13)
        """
        num_queries = self.config.num_queries
        # Gap between query block ranges to simulate non-contiguous access
        gap_between_queries = blocks_per_query  # gap equals to blocks_per_query

        total_blocks_needed = blocks_per_query * num_queries + gap_between_queries * (num_queries - 1)

        if total_blocks_needed > self.config.max_block_num:
            raise ValueError(
                f"Not enough blocks: need {total_blocks_needed} "
                f"(blocks_per_query={blocks_per_query} x num_queries={num_queries} + gaps), "
                f"but max_block_num={self.config.max_block_num}"
            )

        # Generate contiguous blocks for each query with gaps between queries
        query_block_ids_list = []
        start_block = 0
        for query_idx in range(num_queries):
            # Contiguous block_ids within each query
            block_ids = list(range(start_block, start_block + blocks_per_query))
            query_block_ids_list.append(block_ids)
            # Next query starts after current blocks + gap
            start_block += blocks_per_query + gap_between_queries

        return query_block_ids_list

    def synchronize(self):
        """Synchronize device"""
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.synchronize()
        elif paddle.is_compiled_with_xpu():
            paddle.device.xpu.synchronize()

    def run_benchmark(self, ip: str, port: int) -> BenchmarkResult:
        """Run the benchmark and return results"""
        assert self.splitwise_role == "prefill", "only prefill can run benchmark"

        blocks_per_query = min(self.config.blocks_per_transfer, self.config.max_block_num // self.config.num_queries)

        # Generate interleaved block_ids for each query
        query_block_ids_list = self._generate_query_block_ids(blocks_per_query)

        print(f"[Benchmark] Blocks per query: {blocks_per_query}, Queries: {self.config.num_queries}")
        for i, block_ids in enumerate(query_block_ids_list):
            print(f"[Benchmark] Query {i} block_ids: {block_ids[:5]}...{block_ids[-3:] if len(block_ids) > 5 else ''}")

        # Warmup
        print(f"\n[Benchmark] Running {self.config.warmup_iters} warmup iterations...")
        for i in range(self.config.warmup_iters):
            self.write_cache(ip, port, query_block_ids_list)
            self.synchronize()
        print("[Benchmark] Warmup complete")

        # Benchmark
        print(f"[Benchmark] Running {self.config.benchmark_iters} benchmark iterations...")

        self.synchronize()
        start_time = time.perf_counter()

        for i in range(self.config.benchmark_iters):
            self.write_cache(ip, port, query_block_ids_list)

        self.synchronize()
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time

        # Calculate throughput
        # bytes per iteration = blocks_per_query * num_queries * layers * (K + V + K_scale + V_scale)
        bytes_per_iter = blocks_per_query * self.config.num_queries * self.config.num_layers * self.bytes_per_block
        total_bytes = bytes_per_iter * self.config.benchmark_iters
        throughput_gbps = (total_bytes / elapsed_time) / (1024**3)

        result = BenchmarkResult(
            total_bytes=total_bytes,
            elapsed_time=elapsed_time,
            throughput_gbps=throughput_gbps,
            iterations=self.config.benchmark_iters,
            blocks_per_iter=blocks_per_query,
            layers=self.config.num_layers,
            queries_per_layer=self.config.num_queries,
        )

        return result

    def print_results(self, result: BenchmarkResult):
        """Print benchmark results"""
        print("\n" + "=" * 60)
        print("KV Cache Transfer Benchmark Results")
        print("=" * 60)
        print("Configuration:")
        print(f"  - Layers: {result.layers}")
        print(f"  - Queries per layer: {result.queries_per_layer}")
        print(f"  - Blocks per transfer: {result.blocks_per_iter}")
        print(f"  - KV heads: {self.config.kv_num_head}")
        print(f"  - Block size: {self.config.block_size_seq_len}")
        print(f"  - Head dim: {self.config.head_dim}")
        print(f"  - Data type: {self.config.cache_dtype}")
        print("\nResults:")
        print(f"  - Iterations: {result.iterations}")
        print(f"  - Total time: {result.elapsed_time:.4f} s")
        print(f"  - Total data transferred: {self._format_bytes(result.total_bytes)}")
        print(f"  - Throughput: {result.throughput_gbps:.2f} GB/s")
        print(f"  - Avg latency per iter: {(result.elapsed_time / result.iterations) * 1000:.2f} ms")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="KV Cache Transfer Benchmark")
    parser.add_argument(
        "--splitwise_role",
        type=str,
        default="prefill",
        choices=["prefill", "decode"],
        help="Role: prefill (client) or decode (server)",
    )
    parser.add_argument("--decode_ip", type=str, default="127.0.0.1", help="Decode server IP")
    parser.add_argument("--decode_rdma_port", type=int, default=9881, help="RDMA port")
    parser.add_argument("--decode_zmq_port", type=int, default=9882, help="ZMQ port for control")

    # Benchmark config
    parser.add_argument("--num_layers", type=int, default=61, help="Number of transformer layers")
    parser.add_argument("--max_block_num", type=int, default=2000, help="Maximum number of blocks")
    parser.add_argument("--kv_num_head", type=int, default=1, help="Number of KV heads")
    parser.add_argument("--block_size", type=int, default=64, help="Block size (sequence length)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--cache_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "uint8"], help="Cache data type"
    )
    parser.add_argument("--warmup_iters", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--benchmark_iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--blocks_per_transfer", type=int, default=64, help="Number of blocks per transfer")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of queries per layer to simulate")
    parser.add_argument("--device", type=str, default="gpu", help="Device to use (gpu/cpu/xpu)")

    return parser.parse_args()


def run_decode_server(args, config: BenchmarkConfig):
    """Run decode server"""
    context = zmq.Context()
    server_socket = context.socket(zmq.REP)
    server_socket.bind(f"tcp://0.0.0.0:{args.decode_zmq_port}")
    print(f"[Server] ZMQ server started on port {args.decode_zmq_port}")

    _ = KVCacheBenchmark("decode", config, port=args.decode_rdma_port, device=args.device)

    print("[Server] Waiting for benchmark client...")
    while True:
        try:
            obj = server_socket.recv_pyobj()
            if obj.get("msg_type") == "ping":
                server_socket.send_pyobj({"status": "ready"})
            elif obj.get("msg_type") == "benchmark_start":
                print("[Server] Benchmark started by client")
                server_socket.send_pyobj({"status": "ok"})
            elif obj.get("msg_type") == "benchmark_done":
                print("[Server] Benchmark completed")
                server_socket.send_pyobj({"status": "ok"})
                break
            else:
                server_socket.send_pyobj({"status": "unknown"})
        except Exception as e:
            print(f"[Server] Error: {e}")
            break

    print("[Server] Shutting down")


def run_prefill_client(args, config: BenchmarkConfig):
    """Run prefill client (benchmark runner)"""
    context = zmq.Context()
    client_socket = context.socket(zmq.REQ)
    client_socket.connect(f"tcp://{args.decode_ip}:{args.decode_zmq_port}")

    benchmark = KVCacheBenchmark("prefill", config, device=args.device)

    # Connect to decode server
    print(f"[Client] Connecting to decode server at {args.decode_ip}:{args.decode_rdma_port}...")
    benchmark.connect(args.decode_ip, args.decode_rdma_port)

    retry_count = 0
    max_retries = 30
    while not benchmark.is_connected(args.decode_ip, args.decode_rdma_port):
        time.sleep(1)
        retry_count += 1
        if retry_count >= max_retries:
            print("[Client] Failed to connect to decode server")
            return
        print(f"[Client] Waiting for connection... ({retry_count}/{max_retries})")

    print("[Client] Connected to decode server")

    # Ping server
    client_socket.send_pyobj({"msg_type": "ping"})
    reply = client_socket.recv_pyobj()
    if reply.get("status") != "ready":
        print("[Client] Server not ready")
        return

    # Notify benchmark start
    client_socket.send_pyobj({"msg_type": "benchmark_start"})
    client_socket.recv_pyobj()

    # Run benchmark
    result = benchmark.run_benchmark(args.decode_ip, args.decode_rdma_port)
    benchmark.print_results(result)

    # Notify benchmark done
    client_socket.send_pyobj({"msg_type": "benchmark_done"})
    client_socket.recv_pyobj()


def main():
    args = parse_args()

    config = BenchmarkConfig(
        num_layers=args.num_layers,
        max_block_num=args.max_block_num,
        kv_num_head=args.kv_num_head,
        block_size_seq_len=args.block_size,
        head_dim=args.head_dim,
        cache_dtype=args.cache_dtype,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
        blocks_per_transfer=args.blocks_per_transfer,
        num_queries=args.num_queries,
    )

    if args.splitwise_role == "decode":
        run_decode_server(args, config)
    else:
        run_prefill_client(args, config)


if __name__ == "__main__":
    main()
