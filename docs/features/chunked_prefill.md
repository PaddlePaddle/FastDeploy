[简体中文](../zh/features/chunked_prefill.md)

# Chunked Prefill

Chunked Prefill employs a segmentation strategy that breaks down Prefill requests into smaller subtasks, which are then batched together with Decode requests. This approach better balances compute-intensive (Prefill) and memory-intensive (Decode) operations, optimizes GPU resource utilization, reduces computational overhead and memory footprint per Prefill, thereby lowering peak memory usage and avoiding out-of-memory issues.

When Chunked Prefill is enabled, the scheduling policy follows these priority rules:

- **Decode-first processing**: The system prioritizes batching all pending Decode requests to ensure real-time responsiveness for generation tasks.
- **Dynamic chunking**: When Prefill tasks are pending, the system schedules them within the `max_num_batched_tokens` budget (the maximum token capacity per batch). If a Prefill request exceeds the remaining budget, it's automatically split into compliant chunks.

This mechanism effectively reduces Inter-Token Latency while optimizing memory usage, enabling longer input sequences.

For details, refer to the [research paper](https://arxiv.org/pdf/2308.16369).

## Usage & Tuning

When enabling Chunked Prefill in FastDeploy, adjust performance via `max_num_batched_tokens`:

- Smaller values improve **Inter-Token Latency**.
- Larger values improve **Time To First Token**.
- Recommended: `max_num_batched_tokens > 8096`. For small models on high-performance hardware, increase this value for higher throughput.

## Use Cases

1. **Long-context inference** (e.g., 128K context): Reduces memory peaks to avoid OOM errors.
2. **Generation tasks**: Lowers Inter-Token Latency for faster response.
