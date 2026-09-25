[简体中文](../zh/features/infllmv2_attention.md)

# InfLLM-V2 Sparse Attention

The `INFLLMV2_ATTN` backend provides MiniCPM4.1 with two-stage sparse attention during decode and long-context prefill. Stage 1 selects logical cache blocks independently for each request and KV head. Stage 2 maps the logical IDs through `block_tables` and reads the already-written paged K/V cache.

Initial batch-1 prefill uses dense FlashAttention through `dense_len`, then shares one Stage 1 selection across each 128-token query tile. Each sparse tile is split into fully visible selected history blocks and its one or two current blocks. The history partition uses non-causal FlashAttention, the current partition uses causal FlashAttention, and their outputs are combined with the two exact log-sum-exp values. This preserves causal alignment even though Paddle FlashAttention V2 aligns a shorter causal query at the upper left. Short, mixed, and shared chunked-prefill requests retain the dense path while still building the two-scale semantic summaries needed by decode.

## Implementation overview

The decode data flow is:

```text
raw fused QKV
  -> decoder_write_cache_with_rope
       -> post-RoPE / post-QK-norm Q
       -> current K/V committed to the paged cache
  -> infllmv2_update_compressed_k
  -> infllmv2_select_blocks
  -> infllmv2_attention_forward
       -> block_tables[request, logical block] -> physical page
       -> causal sparse softmax
```

Both sparse stages consume the post-RoPE query returned by the writer, and Stage 2 observes the current decode token in the K/V cache. Raw fused QKV is never passed directly to sparse attention.

The sparse prefill data flow is:

```text
raw fused QKV
  -> gqa_rope_write_cache
       -> post-RoPE / post-QK-norm Q/K/V and paged K/V cache
  -> infllmv2_update_compressed_k
  -> dense FlashAttention for [0, dense_len)
  -> one infllmv2_select_blocks call per 128-token query tile
  -> gather selected paged K/V
       -> non-causal FlashAttention over selected history pages
       -> causal FlashAttention over the current one or two pages
       -> exact LSE-weighted output merge
```

The default 4,096-token processing chunk contains 32 query tiles and bounds temporary gathered K/V storage. A final partial tile is supported, including prompts whose length is not block aligned.

For every query token, the request and causal position come from runtime metadata:

```text
request = batch_id_per_token[token]
local_offset = token - cu_seqlens_q[request]
position = seq_lens_decoder[request] + local_offset
```

This keeps requests, KV heads, and page tables isolated under continuous batching. Padding tokens with `batch_id_per_token == -1` do not access the cache.

### Two-scale Stage 1

Stage 1 maintains two K-summary scales:

| Scale | Default window | Default stride | Purpose |
| --- | ---: | ---: | --- |
| Fine | 32 | 16 | Estimate detailed relevance for logical KV blocks |
| Coarse | 128 | 64 | Approximate the log-sum-exp normalizer for each query head |

Each summary is the mean of K vectors in one complete window. A completed window is owned by the physical page containing its final token, which preserves windows crossing page boundaries. With the default `block_size=64`, each physical block and KV head has four fine slots and one coarse slot.

Let `G(g)` be the GQA query-head group associated with KV head `g`, and let `d` be the head dimension. Scores are computed as follows:

```text
LSE[h] = logsumexp_j(dot(Q[h], Kbar_coarse[j]) / sqrt(d))

semantic_score[i, g] = sum(h in G(g)) exp(
    dot(Q[h], Kbar_fine[i]) / sqrt(d) - LSE[h]
)

block_score[b, g] = max(semantic_score[i, g]
                              for fine window i overlapping block b)
```

For the MiniCPM4.1 production shape (`QH=32`, `KVH=2`, `D=128`,
`block_size=64`), the coarse normalizer uses a GQA-tiled split kernel adapted
from the distributed-softmax structure in the OpenBMB CUDA implementation.
One CTA owns `(query token, KV head, coarse split)`. Its 16 half-warps keep the
16 Q heads in registers, cooperatively load one aligned `float4` K tile for up
to 16 coarse windows, and reuse that tile across the whole GQA group. The
per-split online `(max, sum)` values stay distributed across lanes; a second
one-warp-per-query-head kernel performs the rescaled split merge. The split
partials are caller-provided persistent workspaces, so this path has neither a
thread-0 window loop nor a per-call allocation.

The production fine block-score path is tiled across eight adjacent candidate
blocks. One CTA owns `(query token, KV head, eight-block tile)`; its 16
half-warps keep the GQA query heads in registers while up to 33 overlapping
fine K rows are loaded once with aligned `float4` transactions and shared by
the whole group. The same head contributions are then pooled for all eight
candidate blocks. Every score slot, including a partially filled table tail,
is rewritten because the backend reuses this workspace across decode steps.

The first `init_blocks`, the preceding blocks covered by `window_size`, and the current block receive positive-infinity scores and participate in the same top-k; remaining candidates are ranked by `block_score`. With `local_blocks = window_size / block_size`, the sparse-region selection budget is `topk + local_blocks`, so `topk` is not an extra budget beyond the forced blocks. The default selects 96 blocks: 34 forced blocks (one initial, 32 preceding local, and the current block) and at most 62 ordinary dynamic blocks. The output buffer also has to accommodate the select-all behavior below the dense threshold, so its capacity is:

```text
selected_capacity = max(
    topk + window_size / block_size,
    ceil(dense_len / block_size),
)
```

Sparse-region selection uses an exact 64-bit composite radix key: score
descending, then logical block ID ascending. A 256-thread CUB block radix sort
dispatches 1, 2, 4, or 8 candidates per thread and therefore covers up to
2,048 logical blocks (128K tokens at `block_size=64`) in one CTA. This replaces
the previous all-pairs rank computation while preserving deterministic ties.
Requests below `dense_len` bypass sorting and emit all visible blocks directly.
Model configurations exceeding 2,048 candidates fail explicitly.

Stage 1 returns GPU tensors with the following contracts:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `topk_indices` | `[tokens, kv_heads, selected_capacity]` | Logical block IDs per request/query token/KV head; unused slots are `-1` |
| `block_scores` | `[tokens, kv_heads, max_blocks_per_seq]` | Dynamic block scores |
| `selected_counts` | `[tokens, kv_heads]` | Number of selected blocks |

The valid prefix of `topk_indices` must contain strictly increasing, unique logical block IDs within the request's visible range; every remaining slot must be `-1`. Stage 2 uses the first `-1` to terminate the prefix. This is a custom-op precondition guaranteed by Stage 1 in normal serving. FastDeploy's scheduler must likewise provide self-consistent sequence lengths, `cu_seqlens_q`, token-to-request mappings, and `block_tables` page ownership; active requests must not alias writable physical pages. Q/K/V inputs must contain finite values.

MiniCPM4.1 uses GQA, so metadata is produced per KV head rather than duplicated for every query head in its GQA group. Stage 2 shares that KV-head block list across the associated query heads.

### Paged Stage 2

Stage 2 does not assume that logical blocks are contiguous in memory. Every selected block is resolved using:

```text
physical_block = block_tables[request, logical_block]
```

Scaled dot-product attention is then evaluated only for tokens in that physical page that do not exceed the request's current logical `position`. Values in `topk_indices` must therefore be logical block IDs, not physical page IDs.

For the MiniCPM4.1 production shape (`32` query heads, `2` KV heads, `head_dim=128`, `block_size=64`), Stage 2 uses a GQA-tiled FlashDecoding kernel. One 128-thread CTA owns `(query token, KV head, KV split)` and covers all 16 query heads sharing that KV head. It retains a 16-token K/V shared-memory tile: warp 0 evaluates the 16-by-16 QK tensor-core tile, threads 0 through 15 update the per-head online `(max, sum)` state, and all four warps evaluate the probability-times-V tensor-core tiles and update the shared FP32 accumulators. A full 16-token K tile and V tile are copied from the paged cache to shared memory with aligned 16-byte `uint4` transactions; a causal boundary tile uses the explicit scalar masked path. A separate kernel merges the split `(accumulator, max, sum)` partials.

Each KV split contains two selected pages. This yields enough CTAs for batch-1 decode without restoring the old 16-fold K/V global-memory reads. The local CUDA repository instead uses a FlashAttention/CUTE 64-token K/V tile. FastDeploy keeps the smaller tile because a directly ported 64-token paged tile needs 47.3 KiB shared memory and regressed the 128K/batch-4 Stage 2 measurement by 17.1%; the 16-token paged tile uses about 22 KiB while preserving GQA reuse, split-KV, and online softmax. Non-production shapes use the generic correctness path; they do not represent the MiniCPM4.1 performance path.

Each layer backend owns persistent tensors for Stage 1 metadata, final attention output, and the FP32 split `(accumulator, max, sum)` values. All are passed as in-place custom-op inputs and reused across decode steps with the same shape. The CUDA launcher performs no `paddle::empty`, `paddle::full`, or `paddle::zeros` allocation on the Stage 2 hot path.

## Configuration and activation

Select the backend explicitly:

```shell
export FD_ATTENTION_BACKEND=INFLLMV2_ATTN
```

Sparse parameters are read first from `sparse_config` in the model's `config.json`, then from same-named top-level model fields, and finally from the MiniCPM4.1 defaults. To override them, add:

```json
{
  "sparse_config": {
    "kernel_size": 32,
    "kernel_stride": 16,
    "topk": 64,
    "dense_len": 8192,
    "init_blocks": 1,
    "window_size": 2048,
    "sparse_prefill": true,
    "prefill_query_chunk_size": 4096
  }
}
```

| Parameter | Default | Constraint |
| --- | ---: | --- |
| `block_size` | 64 | The `--block-size` server option; divisible by both `kernel_stride` and `4 * kernel_stride` |
| `kernel_size` | 32 | Positive integer |
| `kernel_stride` | 16 | Positive integer |
| `topk` | 64 | Positive integer |
| `dense_len` | 8192 | At least `4 * kernel_size` |
| `init_blocks` | 1 | Non-negative integer |
| `window_size` | 2048 | A non-negative multiple of `block_size` |
| `sparse_prefill` | `true` | Enables the model-specific initial-prefill sparse path |
| `prefill_query_chunk_size` | 4096 | Positive multiple of 128 |

`init_blocks` must be smaller than `topk` so the initial blocks and the inclusive local set fit in the output capacity.

The following example starts a single-GPU server for a 32K context:

```shell
export CUDA_VISIBLE_DEVICES=0
export FD_ATTENTION_BACKEND=INFLLMV2_ATTN
export MODEL_PATH=/path/to/MiniCPM4.1-8B

.venv/bin/python -m fastdeploy.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name MiniCPM4.1-8B \
  --port 8180 --metrics-port 8181 \
  --engine-worker-queue-port 8182 --cache-queue-port 8183 \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --block-size 64 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.5 \
  --no-enable-prefix-caching \
  --graph-optimization-config '{"use_cudagraph": false}'
```

## Limitations and memory cost

- The custom operators support FP32, FP16, and BF16 on NVIDIA CUDA devices with compute capability 8.0 or newer (SM80+).
- The paged K/V cache must be unquantized and rank 4. Quantized KV cache fails explicitly.
- The number of query heads must be divisible by the number of KV heads, and `head_dim` must be in `[1, 256]`.
- Tensor parallelism must not replicate KV heads: `tensor_parallel_size` must be no larger than every layer's global KV-head count. Replica-spanning Stage 1 score reduction is not implemented yet, so unsupported configurations fail explicitly.
- Sparse decode requires semantic K summaries initialized by prefill on the same paged K cache. P/D disaggregation does not currently transfer these summaries and must not enter sparse decode without them.
- Resetting or replacing the paged K cache must also reset its semantic summaries.
- Semantic summaries do not yet participate in CUDA Graph cache replacement, so CUDA Graph must currently be disabled.
- Speculative decoding/MTP is rejected until every proposed token participates in Stage 1 metadata and semantic-cache updates.
- Sparse prefill currently activates only for an initial causal batch-1 request with the MiniCPM4.1 production shape (`QH=32`, `KVH=2`, `D=128`, `block_size=64`) and FP16/BF16 cache. Other layouts fall back to dense prefill.
- The 32K acceptance run uses one scheduler-level 32K prefill request (internally processed in 4,096-token sparse batches) and therefore does not exercise scheduler-level multi-chunk prefill. With `--max-num-batched-tokens 8192 --enable-chunked-prefill`, both the existing FlashAttention service and InfLLM-V2 service abort in the third 8K prefill chunk on an uninitialized Paddle `optional<T>::get()`. This isolates the failure to the shared chunked-prefill path rather than sparse Stage 1/Stage 2. This implementation does not change that shared path; both comparison variants use `--max-num-batched-tokens 32768`.

The theoretical additional semantic-cache storage per layer and physical block is:

```text
dtype_bytes * kv_heads * head_dim * (
    block_size / kernel_stride
    + block_size / (4 * kernel_stride)
)
```

For MiniCPM4.1 BF16 with `kv_heads=2`, `head_dim=128`, and the default windows, this is 2,560 bytes/block/layer. This is a shape-derived static cost, not the measured INF-3 peak memory result.

## Build and regression

The CUDA custom operators must be rebuilt before operator tests are run. This example uses the project's `.venv`; adjust `CUDA_HOME` and the compute-capability list for the target machine:

```shell
export CUDA_HOME=/path/to/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PWD}"

MAX_JOBS=32 FD_BUILD_RESUME=1 \
  bash build.sh 0 "${PWD}/.venv/bin/python" false "[86]"
```

Run the focused regressions with:

```shell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/model_executor/test_infllmv2_attention_backend.py \
  tests/operators/test_infllmv2_attention_forward.py
```

The backend tests cover writer/Stage 1/Stage 2 ordering, post-RoPE query use, and semantic-cache lifetime. GPU operator tests cover cross-page semantic windows, per-request/per-KV-head selection, paged mapping, the short-context closed loop, and invalid metadata. This feature does not require documentation tests and must not modify `tests/model_executor/test_minicpm41.py`.

## Dense/sparse long-context comparison

`benchmarks/benchmark_infllmv2.py` has seven subcommands:

- `selector` inspects a raw selector trace and writes an unbound hit-rate summary. This summary is for inspection only and cannot be attached to a timed sparse run.
- `sparse-diagnostic` runs the sparse workload once against a trace-enabled server, automatically forwards `--no-warmup` to `benchmark_serving.py`, and binds the newly generated trace to the exact prompt-token workload in a selector diagnostic JSON.
- `run` delegates throughput, TTFT, and TPOT collection to `benchmark_serving.py` while sampling memory on one GPU. A timed sparse run requires the bound JSON through `--selector-diagnostic`.
- `report` produces a comparison JSON and Markdown table only when dense and sparse workload fingerprints match exactly and both runs report the same physical GPU UUID.
- `operators` uses CUDA events to time compressed-K update, Stage 1, Stage 2, and the complete sparse decode operator chain. Its default matrix is 32K/concurrency 1 and 128K/concurrency 4.
- `prefill` compares the complete dense and sparse Paddle prefill paths with CUDA events.
- `cuda-impl` runs the equivalent decode workload against a local `infllm_v2` PyTorch extension checkout.

Run the calibrated operator matrix independently from service timing:

```shell
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_infllmv2.py operators \
  --gpu-index 7 \
  --scenario 32768:1 \
  --scenario 131072:4 \
  --output runs/bench/infllmv2/operators.json
```

The JSON records mean/median/P10/P90/P99 CUDA time, host wall time, source hashes, the selected-token fraction, and persistent workspace bytes for each scenario. It also verifies that all four Stage 2 outputs alias the caller-provided buffers. This synthetic benchmark isolates kernel costs; it is not a substitute for TTFT/TPOT service comparison.

Sparse prefill has a dedicated complete-chain benchmark:

```shell
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv/bin/python \
  benchmarks/benchmark_infllmv2.py prefill \
  --device gpu:0 \
  --context-length 16384 \
  --context-length 32768 \
  --output runs/bench/infllmv2/sparse_prefill.json
```

The sparse timing includes compressed-K update, the dense prefix, Stage 1, paged-cache gather, both FlashAttention partitions, and the LSE merge. On an NVIDIA RTX A6000 with BF16 and batch 1, the 2026-08-18 run measured 90.300 ms dense versus 46.005 ms sparse at 32K, or **1.963x**. At 16K it measured 21.256 ms versus 18.654 ms, or 1.139x, showing that the optimization is intended for long contexts. The traceable result is `runs/bench/infllmv2/sparse_prefill_20260818.json`.

### 32K/batch-1 theoretical ceiling

With defaults, sparse Stage 2 selects 96 pages of 64 tokens. At 32K this is 6,144 of 32,768 visible tokens, or 18.75%. Even under an ideal bandwidth-only model, Stage 2's token-reduction ceiling is therefore `32768 / 6144 = 5.333x`. End-to-end decode cannot exceed that number and is bounded more tightly by Amdahl's law:

```text
speedup <= 1 / ((1 - f_stage2) + f_stage2 / 5.333)
```

where `f_stage2` is dense decode's Stage 2-attention time fraction. Stage 1, compressed-K update, split merge, launch overhead, MLP/MoE, and batch-1 GPU underfill are outside the idealized reduction and lower the observed gain. At 128K the selected-token ratio is lower, but Stage 1 scans more semantic windows, so the token ratio alone must not be reported as measured speedup.

### Selector hit rate

Hit rate is the micro-averaged dense-reference block recall across all request/query-token/KV-head samples:

```text
sum(|selected_blocks ∩ reference_blocks|) / sum(|reference_blocks|)
```

`reference_blocks` should come from exact dense attention using the same post-RoPE query and paged K cache, aggregated over each GQA group to KV-head granularity. The raw format is:

```json
{
  "samples": [
    {
      "selected_blocks": [0, 12, 37],
      "reference_blocks": [0, 9, 37]
    }
  ]
}
```

Block IDs must be non-negative and unique within an array, and `reference_blocks` must not be empty. These numbers only demonstrate the JSON format; they are not measured model results.

The backend can produce these samples directly in a separate diagnostic run. Set the trace variables before starting a dedicated `INFLLMV2_ATTN` diagnostic server:

```shell
export FD_INFLLMV2_SELECTOR_TRACE_PATH="${PWD}/runs/bench/infllmv2/selector_samples.json"
export FD_INFLLMV2_SELECTOR_TRACE_RANK=0
export FD_INFLLMV2_SELECTOR_TRACE_LAYER=0
export FD_INFLLMV2_SELECTOR_TRACE_MAX_SAMPLES=16
```

CUDA Graph must be disabled while tracing. After Stage 1, the backend computes an FP32 dense softmax from the same post-RoPE query, `block_tables`, and already-written paged K cache. It sums logical-block attention mass over the query heads sharing each KV head and uses the highest-mass blocks, with the same cardinality as `selected_count`, as the reference. Tracing introduces GPU synchronization, device-to-host copies, and dense-attention work, so diagnostic latency is never treated as a performance result. The trace path must not exist before the diagnostic, and the server refuses to overwrite it.

```shell
mkdir -p runs/bench/infllmv2

.venv/bin/python benchmarks/benchmark_infllmv2.py sparse-diagnostic \
  --base-url http://127.0.0.1:8180 \
  --model MiniCPM4.1-8B \
  --tokenizer "${MODEL_PATH}" \
  --input-len 32768 \
  --output-len 128 \
  --num-prompts 4 \
  --max-concurrency 1 \
  --trace-path runs/bench/infllmv2/selector_samples.json \
  --output runs/bench/infllmv2/selector_diagnostic.json
```

`sparse-diagnostic` automatically disables the normal warm-up request by passing `--no-warmup`; there is no separate warm-up result to report. It records the exact generated prompt-token digest and workload fingerprint together with the trace provenance. The later timed sparse `run` rejects a diagnostic whose workload differs.

The raw trace can also be summarized for standalone inspection, but this unbound output is not accepted by `run`:

```shell
.venv/bin/python benchmarks/benchmark_infllmv2.py selector \
  --samples runs/bench/infllmv2/selector_samples.json \
  --output runs/bench/infllmv2/selector_summary.json
```

### Fixed-workload runs

After the diagnostic finishes, stop the trace-enabled sparse server completely. Keep the raw trace file because the bound diagnostic verifies it by path and SHA256, but disable tracing and restart the server before any timing measurement:

```shell
unset FD_INFLLMV2_SELECTOR_TRACE_PATH
```

The diagnostic server and timed server must be separate processes. Confirm that the timed `INFLLMV2_ATTN` server starts without the trace-path variable; trace instrumentation must remain disabled for the complete timed run.

For the timed comparison, first start a dense server with `FD_ATTENTION_BACKEND=FLASH_ATTN`, then run:

```shell
.venv/bin/python benchmarks/benchmark_infllmv2.py run \
  --variant dense \
  --base-url http://127.0.0.1:8180 \
  --model MiniCPM4.1-8B \
  --tokenizer "${MODEL_PATH}" \
  --gpu-index 0 \
  --input-len 32768 \
  --output-len 128 \
  --num-prompts 4 \
  --max-concurrency 1 \
  --output runs/bench/infllmv2/dense.json
```

Stop the dense server. Start the untraced `INFLLMV2_ATTN` server on the same idle GPU with otherwise identical server arguments, then run:

```shell
.venv/bin/python benchmarks/benchmark_infllmv2.py run \
  --variant sparse \
  --base-url http://127.0.0.1:8180 \
  --model MiniCPM4.1-8B \
  --tokenizer "${MODEL_PATH}" \
  --gpu-index 0 \
  --input-len 32768 \
  --output-len 128 \
  --num-prompts 4 \
  --max-concurrency 1 \
  --selector-diagnostic runs/bench/infllmv2/selector_diagnostic.json \
  --output runs/bench/infllmv2/sparse.json

.venv/bin/python benchmarks/benchmark_infllmv2.py report \
  --dense-result runs/bench/infllmv2/dense.json \
  --sparse-result runs/bench/infllmv2/sparse.json \
  --output runs/bench/infllmv2/report.json
```

The workload arguments supplied to `sparse-diagnostic`, dense `run`, and sparse `run` must be identical, including model, tokenizer, endpoint, seed, prompt count, input/output lengths, request rate, and concurrency. The script verifies the generated prompt tokens and rejects mismatched fingerprints.

`--gpu-index` is the physical index consumed by `nvidia-smi --id`. If the server uses `CUDA_VISIBLE_DEVICES=7`, pass `--gpu-index 7`, not the process-local index 0. Keep other workloads off that GPU and restart the server completely between variants.

In each `run` result, `gpu_memory.gpu_uuid` identifies the physical device, `baseline_mib` is the already-started server's memory immediately before requests, `peak_mib` is total used GPU memory at peak, and `peak_delta_mib` is their difference. `report` rejects dense and sparse inputs with different GPU UUIDs and preserves the UUID plus baseline, peak, and peak-delta values in the report. An acceptance report must retain all of them.

### Retained P0/P2 checkpoint results

The final CUDA-event matrix is stored in `runs/bench/infllmv2/tmp/p3_operator_matrix.json`. It uses the MiniCPM4.1 production shape (`QH=32`, `KVH=2`, `D=128`, BF16), 50 warm-up iterations, and 200 measured iterations on an NVIDIA RTX A6000. Values below are median device times.

| Scenario | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | Complete sparse chain (us) | Stage 2 persistent bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 49.152 | 191.488 | 111.616 | 380.928 | 1,073,152 |
| 128K / concurrency 4 | 49.152 | 1,662.976 | 215.040 | 1,939.968 | 4,292,608 |

The pre-P0/P2 Stage 2 baseline is retained in `runs/bench/infllmv2/tmp/p0_p2_operator_baseline.json`. Comparing the same shapes on the same GPU gives:

| Scenario | Pre-P0 Stage 2 (us) | GQA-tiled Stage 2 (us) | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 135.168 | 111.616 | 1.211x | 17.42% |
| 128K / concurrency 4 | 367.616 | 215.040 | 1.710x | 41.50% |

The dedicated P0 checkpoint itself measured 112.144 us and 215.040 us for Stage 2, or 1.205x and 1.710x over the retained baseline. The 32K difference from the final 111.616 us value is run-to-run variation.

The matrix also verifies that the output and all three FP32 partial tensors alias caller-provided storage. These byte counts are per layer-backend instance: Stage 2 retains 1,073,152 bytes at concurrency 1 and 4,292,608 bytes at concurrency 4. Each instance allocates on shape change rather than on every layer invocation; for this 32-layer checkpoint, the corresponding totals are 32.75 MiB and 131.00 MiB. The P2 checkpoint measured 112.640 us for 32K/1 Stage 2 versus 112.144 us at the P0 checkpoint (+0.44%), while the full chain changed from 381.952 us to 380.928 us (-0.27%); 128K/4 Stage 2 remained 215.040 us. This is within run-to-run noise and shows no hot-path regression from in-place persistence. The P0 and P2 checkpoints are preserved separately as `p0_gqa_tiled_operator.json` and `p2_persistent_workspace_operator.json` in the same temporary-results directory.

The checkpoint used here declares both `max_position_embeddings` and `rope_scaling.original_max_position_embeddings` as 65,536. A 128K service request would violate that model contract, so the required 128K/concurrency-greater-than-one control is intentionally a synthetic operator comparison and is not reported as an end-to-end model result.

### Stage 1 selector optimizations

The subsequent Stage 1 GQA-tiled coarse-LSE and exact radix-TopK measurements
are stored separately in
`runs/bench/infllmv2/tmp/stage1_p1_gqa_tiled_result.json` and
`runs/bench/infllmv2/tmp/stage1_p2_radix_topk_result.json`. The combined CUDA
event result is `stage1_p1_p2_optimized.json`; Nsight Systems traces and SQLite
exports in the same directory provide the per-kernel attribution. These runs
use the same RTX A6000 and BF16 production shape, with 20 warm-ups and 100
measured iterations for CUDA-event medians.

| Scenario | Baseline Stage 1 (us) | Optimized Stage 1 (us) | Stage 1 speedup | Baseline chain (us) | Optimized chain (us) | Chain speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 191.488 | 82.944 | 2.309x | 381.952 | 246.784 | 1.548x |
| 128K / concurrency 4 | 1,659.904 | 820.224 | 2.024x | 1,935.360 | 1,052.672 | 1.839x |

| Scenario | Baseline coarse LSE (us) | GQA-tiled coarse LSE (us) | Speedup | Baseline TopK (us) | Radix TopK (us) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 111.155 | 7.635 | 14.558x | 34.643 | 13.978 | 2.478x |
| 128K / concurrency 4 | 479.877 | 22.688 | 21.151x | 509.055 | 22.579 | 22.545x |

At that checkpoint, block-score calculation was the dominant Stage 1 kernel at
128K/concurrency 4 (773.271 us, about 94% of Stage 1) and was therefore the
next selector bottleneck. Its isolated extension passed 10 SM86 GPU
correctness cases, including a production-shape BF16 semantic reference and
2,048-candidate exact-tie/padding cases; the backend regression suite passed 39
cases. After that checkpoint's package rebuild, the packaged SM86 operator
suite passed all 15 cases. The block-score bottleneck is addressed in the next
section.

### GQA-tiled block-score and final Stage 2 refinements

The follow-up implementation was informed by the split-KV and distributed
softmax structure in the official OpenBMB repository, cloned under
`runs/bench/infllmv2/tmp/infllmv2_cuda_impl` at commit
`93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6`. FastDeploy retains its own paged
cache, persistent-workspace, and in-place custom-op contracts.

The final isolated CUDA-event matrix is
`runs/bench/infllmv2/tmp/stage1_p3_gqa_blockscore/operator_optimized_halfwarp.json`.
It uses the same RTX A6000, BF16 production shape, 20 warm-ups, and 100 measured
iterations as the preceding selector comparison.

| Scenario | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | Complete sparse chain (us) |
| --- | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 49.152 | 48.128 | 59.392 | 149.504 |
| 128K / concurrency 4 | 49.152 | 131.072 | 124.928 | 317.440 |

Relative to the retained P0/P2 matrix, the complete chain falls from 380.928
to 149.504 us at 32K/1 (2.548x, 60.75% less time) and from 1,939.968 to
317.440 us at 128K/4 (6.111x, 83.64% less time). Relative to the immediately
preceding coarse-LSE/radix-TopK checkpoint, Stage 1 is 1.723x faster at 32K/1
and 6.258x faster at 128K/4; Stage 2 is 1.879x and 1.787x faster.

Nsight Systems reports, SQLite exports, and generated kernel-summary CSV files
are in the same temporary directory. The per-kernel averages below compare the
preceding checkpoint with the final implementation.

| Kernel | 32K before/final (us) | Speedup | 128K/4 before/final (us) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Fine block-score | 58.585 / 13.370 | 4.382x | 773.271 / 82.341 | 9.391x |
| Stage 2 partial | 70.386 / 40.153 | 1.753x | 152.721 / 95.404 | 1.601x |
| Stage 2 merge | 37.548 / 15.059 | 2.493x | 72.025 / 25.997 | 2.770x |

The current-source isolated extension passed 14/14 GPU cases, including
FP32/FP16/BF16 paged Stage 2, production BF16 GQA Stage 1/2, exact 2,048-way
ties, reusable-buffer tails, and invalid contracts. The production package and
matched service result are reported after the full rebuild below.

For the supported 32K service workload, the retained P0/P2 checkpoint report
`runs/bench/infllmv2/tmp/p3_report_32k_current.json` compares dense and sparse
servers launched from that same packaged build with identical service
parameters. It predates the Stage 1 selector optimizations above. Both inputs
have workload ID
`b0fd4419f876adfa40e4d92a6e46d56c95c0f5f37e0539abd1a32c8208f6c078`, the
same prompt-token hash, and the same physical GPU UUID; the report command
validated these fields.

| Metric | Dense | P0/P2 checkpoint sparse | Sparse / Dense |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 64.4531% (990 / 1536) | — |
| Request throughput (req/s) | 0.092799 | 0.084076 | 0.906x |
| Output throughput (token/s) | 11.878301 | 10.761664 | 0.906x |
| Total-token throughput (token/s) | 3052.723423 | 2765.747673 | 0.906x |
| Mean / median / P99 TTFT (ms) | 7368.456 / 7362.350 / 7399.758 | 7404.749 / 7389.901 / 7467.811 | 1.005x / 1.004x / 1.009x |
| Mean / median / P99 TPOT (ms) | 26.742 / 26.744 / 26.794 | 35.252 / 35.408 / 35.514 | 1.318x / 1.324x / 1.325x |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 0.996x / 0.997x / 1.000x |

The service workload contains four deterministic random-token requests with input 32768, output 128, concurrency 1, request rate `inf`, and seed 2026. It uses TP1, BF16, `block_size=64`, default sparse configuration `32/16/64/8192/1/2048`, no prefix cache, no CUDA Graph, and one 32K prefill chunk. The environment is an NVIDIA RTX A6000 48 GiB (UUID `GPU-56bb1092-c218-9cd3-ad85-9a2735522d28`), CUDA toolkit 12.8, Paddle 3.3.1 built with CUDA 12.6, and Python 3.12.13.

That retained service result is not an end-to-end measurement of the new Stage
1 kernels: its sparse output throughput is 0.906x dense and mean TPOT is 1.318x
dense. The current-source operator chain is 35.39% shorter at 32K/1 and 45.61%
shorter at 128K/4. At that checkpoint, a packaged server rebuild and matched
service rerun were still required; the completed rerun is reported below. The older
`runs/bench/infllmv2/report.json` remains a still earlier historical result
(0.311x dense output throughput and 7.945x dense mean TPOT) and must not be
presented as the current kernel result.

### Historical post-Stage1 packaged service checkpoint

The full `fastdeploy_ops` package was rebuilt from the then-current sources
before this checkpoint. The installed library had SHA-256
`cb37a51292ed2f92139a199f748a1e8fb014d6c7774284cfdc6bfe535ed2d5e4`, and
the packaged GPU operator regression passed 15/15 cases. Build, correctness,
server, and benchmark logs are retained under
`runs/bench/infllmv2/tmp/retest_stage1_20260813/`.

The validated comparison is
`runs/bench/infllmv2/tmp/retest_stage1_20260813/report_32k.json`. Dense and
sparse use the same workload ID and prompt-token hash shown above, the same
physical GPU UUID, and completely restarted servers. The timed sparse server
did not have selector tracing enabled.

| Metric | Dense | Post-Stage1 checkpoint sparse | Sparse / Dense |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 59.2448% (910 / 1536) | — |
| Request throughput (req/s) | 0.092539 | 0.087938 | 0.950x |
| Output throughput (token/s) | 11.845043 | 11.256072 | 0.950x |
| Total-token throughput (token/s) | 3044.176118 | 2892.810553 | 0.950x |
| Mean / median / P99 TTFT (ms) | 7403.585 / 7401.545 / 7445.184 | 7414.495 / 7413.212 / 7430.486 | 1.001x / 1.002x / 0.998x |
| Mean / median / P99 TPOT (ms) | 26.710 / 26.717 / 26.747 | 31.071 / 31.058 / 31.133 | 1.163x / 1.162x / 1.164x |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 0.996x / 0.997x / 1.000x |

Relative to the retained P0/P2 sparse checkpoint, mean sparse TPOT fell from
35.252 ms to 31.071 ms (11.86% reduction, 1.135x speedup) and sparse output
throughput rose from 10.761664 to 11.256072 token/s (+4.59%). The repeated
dense mean TPOT changed by only -0.12%, so the sparse gain is not explained by
a faster dense baseline. Selector diagnostics follow generated decode
trajectories; the diagnostic runs share prompt fingerprints but not generated
token identities, so the current 59.24% and historical 64.45% hit rates should
not be treated as a controlled selector-quality delta.

At this historical checkpoint, sparse improved but did not cross dense at
32K/batch 1: its mean TPOT
is still 4.361 ms higher (1.163x dense), and output throughput is 0.950x dense.
The 5.333x token-reduction ceiling applies only to Stage 2 K/V work. Dense
FlashAttention is already fused and efficient, while sparse decode additionally
pays compressed-K update, block-score/coarse-LSE/TopK selection, split merge,
kernel-launch overhead across 32 layers, and the same MLP/MoE cost. Batch 1 also
underfills the GPU. These fixed costs exceed the saved Stage 2 K/V traffic in
this checkpoint, so sparsity alone does not imply an end-to-end speedup.

### Final GQA-tiled packaged validation and service rerun

After the block-score and final Stage 2 changes, the complete SM86 production
package was rebuilt again. The repository-packaged
`fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` has
SHA-256
`368174499c47eb991595616988f1c2a65bd6384f8cf8353d366511784883bb94`.
The backend regression passed 39/39 cases and the packaged GPU operator suite
passed 15/15 cases. The successful build and test logs are under
`runs/bench/infllmv2/tmp/stage1_p3_gqa_blockscore/full_build/`.

The packaged CUDA-event rerun is
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/operators_packaged.json`.
It records the final source hashes, verifies all four Stage 2 output aliases,
and gives these device medians:

| Scenario | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | Complete sparse chain (us) |
| --- | ---: | ---: | ---: | ---: |
| 32K / concurrency 1 | 49.152 | 37.888 | 59.392 | 149.504 |
| 128K / concurrency 4 | 49.152 | 131.072 | 125.952 | 316.416 |

Against the retained P0/P2 chain this is a 2.548x speedup at 32K/1 and a
6.131x speedup at 128K/4. Against the pre-P0 Stage 2 baseline, final packaged
Stage 2 is 2.276x and 2.919x faster, respectively. The small 128K difference
between the isolated 124.928 us/317.440 us values above and the packaged
125.952 us/316.416 us values is run-to-run variation.

The final matched serving artifacts are under
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/`. They retain the same
workload ID, prompt-token hash, physical GPU UUID, and server arguments as the
historical comparisons. The sparse service was independently restarted and
measured twice; neither timed run enabled selector tracing.

| Metric | Dense | Final sparse run 1 | Final sparse repeat |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 57.5521% (884 / 1536) | Same bound diagnostic |
| Request throughput (req/s) | 0.092431 | 0.073803 (0.798x) | 0.074985 (0.811x) |
| Output throughput (token/s) | 11.831154 | 9.446807 (0.798x) | 9.598114 (0.811x) |
| Total-token throughput (token/s) | 3040.606665 | 2427.829371 (0.798x) | 2466.715343 (0.811x) |
| Mean / median / P99 TTFT (ms) | 7394.920 / 7395.463 / 7436.516 | 11214.213 / 11260.205 / 11902.204 | 10500.386 / 10554.044 / 10999.020 |
| Mean / median / P99 TPOT (ms) | 26.876 / 26.880 / 27.012 | 18.315 / 19.624 / 25.340 | 22.248 / 22.148 / 24.921 |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 19417 / 24345 / 4928 |

Both sparse runs reduce client-observed mean TPOT: by 31.85% (1.467x) and
17.22% (1.208x). This is the first matched 32K result in this implementation
where sparse TPOT is lower than dense TPOT. It is not a universal end-to-end
speedup: sparse TTFT is 42.0%-51.6% higher and the 128-token output throughput
is only 0.798x-0.811x dense.

Streaming is also bursty in the final sparse runs: 45.08% and 37.20% of ITLs
are below 1 ms, versus 0% for dense. The benchmark's clean rate after excluding
those bursts is 37.21 token/s for dense and 30.04/28.26 token/s for sparse.
Consequently, the report treats the two lower TPOT measurements as reproducible
client-observed latency, reports their range, and does not reinterpret the
single best 1.467x value as stable kernel throughput. The direct CUDA-event and
Nsight results above remain the kernel-level evidence.

This resolves why sparse is theoretically attractive without assuming it must
win every metric: it reads only 18.75% of the dense Stage 2 tokens at 32K, but
still pays selector, merge, launch, prefill/TTFT, and non-attention model costs.
The complete raw results and this interpretation are summarized in
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/summary.md`.

### CUDA-restored baseline rerun (2026-08-16)

The checkpoint sections above are retained as optimization history. The
current-source baseline at that checkpoint is
`runs/bench/infllmv2/service_gate_65k8_20260816/summary.md`; it was produced
after a fresh SM86 rebuild and does not reuse those historical conclusions.
The build output and loaded repository library are byte-identical, size
1,040,001,400 bytes and SHA-256
`05e5b67e43d0da09bcbb884ae5b78fba23b8a3a99e31be69a3216c7a5d00e2c7`.
The focused current-package regression passed 24/24 tests, including
cross-page compressed-K, Stage 1 score/Top-K, paged Stage 2, and independent
NumPy/Paddle references.

On the model-valid 65,344-token, batch-8 BF16 workload (`QH=32`, `KVH=2`,
`D=128`, 96 selected pages), CUDA-event medians are:

| Implementation | Stage 1 + Top-K (us) | Stage 2 (us) | Sparse chain (us) | Dense baseline (us) |
| --- | ---: | ---: | ---: | ---: |
| FastDeploy | 349.184 | 318.464 | 650.240 | 785.408 |
| local `infllmv2_cuda_impl` | 318.464 | 208.896 | 573.440 | 1,563.648 |

FastDeploy passes its own operator gate by 1.2079x. Its Stage 1 + Top-K,
Stage 2, and sparse chain are respectively 1.0965x, 1.5245x, and 1.1339x
slower than the local implementation. The dense columns are not comparable
across implementations because their dense references differ. The synthetic
page-aligned operator rate is 98,304 / 1,045,504 = 9.40255%. In the bound
service trace, partial causal pages make the actual rate 97,332 / 1,045,556 =
9.30911%, versus a 9.40208% full-page preset (-0.09296 percentage point).

Nsight identifies Stage 2 main as the largest sparse kernel (286.205 us),
followed by fine block-score (249.816 us). A single 64-token shared-page Stage
2 experiment was rejected: although it improved 32K/batch-1, it regressed
128K/batch-4 Stage 2 by 17.1%, and Nsight attributed the regression to the
47.3 KiB shared-memory tile reducing occupancy. The retained 16-token tile
preserves GQA K/V sharing, split-KV, and online softmax.

Dense and sparse services were then completely restarted and measured with
the same 65,344-input/128-output, eight-request, concurrency-8 workload. The
validated workload ID is
`4068415f92b013c216341f8461e29f42c979aa0aa76070b9677844817d68a1fd`.
Sparse did not beat dense end to end: output throughput was 5.439 versus 5.531
token/s (0.983x), mean TTFT was 108,132 versus 105,124 ms (1.029x), and mean
TPOT was 628.660 versus 627.799 ms (1.001x). Selector micro-recall was 53.84%
(827/1,536).

### Vectorized paged-K/V rerun (2026-08-17)

The authoritative current-source evidence is
`runs/bench/infllmv2/service_gate_vector_load_20260817/summary.md`. The SM86
package was rebuilt again. The build output and repository-loaded library are
byte-identical, size 1,040,001,400 bytes and SHA-256
`1370f08391426352349d420b9268a769d8960615d32a717f1bb66a349dca59dc`.
The source header SHA-256 is
`883dbc1f5bfbfa8748bb83b3436766f7681d034a26de2e1811661aa738ccd6ae`.
The focused regression again passed 24/24 tests.

This round changed only full-tile paged K/V movement in the production Stage
2 kernel: the scalar global-to-shared loop became aligned 16-byte `uint4`
loads/stores. Partial causal tiles keep the explicit scalar masking path.
Stage 1, CUB radix Top-K, the 16-token data layout, four-warp tensor-core
parallel structure, two-pages-per-split policy, online softmax, and split
combine remain unchanged. This differs from the local repository's 64-token
FlashAttention/CUTE tile only where paged-cache occupancy evidence requires it;
no simpler scalar attention algorithm was substituted.

With the same 65,344-token, batch-8 BF16 workload and 20 warmups plus 100 CUDA
event measurements, the new medians are:

| Implementation | Stage 1 + Top-K (us) | Stage 2 (us) | Sparse chain (us) | Dense baseline (us) |
| --- | ---: | ---: | ---: | ---: |
| FastDeploy | 329.728 | 163.840 | 507.424 | 784.384 |
| local `infllmv2_cuda_impl` | 304.112 | 206.848 | 547.840 | 1,553.408 |

Relative to the 2026-08-16 FastDeploy baseline, Stage 2 is 1.9438x faster and
the sparse chain is 21.96% lower. FastDeploy now passes its own dense gate by
1.5458x. Against the freshly rerun local implementation, FastDeploy Stage 2 is
1.2625x faster and its sparse chain is 1.0796x faster, while its Stage 1 +
Top-K remains 1.0842x slower. Dense references remain framework-specific and
are not compared across implementations.

Nsight Systems measures the Stage 2 main kernel at 145.918 us, down from
286.205 us. Fine block-score is now the largest sparse kernel at 255.564 us;
the next optimization round must therefore target Stage 1 rather than tune
Stage 2 block counts. The operator effective rate is 98,304 / 1,045,504 =
9.40255%. The current bound service trace is 97,332 / 1,045,556 = 9.30911%,
versus a 9.40208% full-page preset (-0.09296 percentage point).

Dense and untraced sparse services were fully restarted with the same validated
workload ID. The complete service still did not beat dense: sparse output
throughput was 4.832 versus 4.915 token/s (0.983x), mean TTFT was 133,586 versus
131,160 ms (1.018x), and mean TPOT was 614.727 versus 605.495 ms (1.015x).
Selector micro-recall was 53.39% (820/1,536). The operator gain is real, but it
does not justify a universal end-to-end speedup claim for this long-prefill,
128-token-output workload.

## W4A16 attention and serving E2E record (2026-08-25)

This section uses one convention throughout: latency speedup is
`FlashAttention latency / InfLLM-v2 latency`, while throughput speedup is
`InfLLM-v2 throughput / FlashAttention throughput`. Values above 1 mean that
InfLLM-v2 is faster. Attention-only measurements must not be confused with
service E2E, which also includes W4A16 Linear layers, MLPs, normalization,
sampling, scheduling, and API overhead.

### Current-source standalone attention measurements

The isolated setup used one RTX A6000 (SM86), the MiniCPM4.1 production shape
`QH=32`, `KVH=2`, `D=128`, `block_size=64`, batch 1, and BF16 Q/K/V plus BF16
KV cache. Prefill used 20 warmups and 100 CUDA-event samples; decode used 50
warmups and 200 samples.

| Phase | Context | FlashAttention median | InfLLM-v2 complete-chain median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Prefill attention | 16,384 | 20.824 ms | 18.514 ms | 1.125x |
| Prefill attention | 32,768 | 88.638 ms | 45.810 ms | 1.935x |
| Decode attention | 32,768 | 62.464 us | 181.248 us | 0.345x |

The 32K decode chain comprises a 16.384 us summary update, 95.232 us Stage 1,
and 58.368 us Stage 2. Stage 2 alone is 1.070x faster than dense attention, but
the complete chain is only 0.345x, or 2.902x slower. A Stage-2-only number is
therefore not a decode-attention speedup. See the raw
[prefill JSON](../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/attention_prefill_16k_32k_b1.json)
and [decode JSON](../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/attention_decode_32k_b1.json).

### W4A16 serving definition and current status

W4A16 means INT4 Linear weights with BF16 activations. Attention Q/K/V and KV
cache remain BF16. The only intended comparison variable was the attention
backend; both services otherwise used TP1, `max_model_len=65536`,
`block_size=64`, `max_num_seqs=1`, `max_num_batched_tokens=32768`, with prefix
caching, chunked prefill, and CUDA Graph disabled. The workload was four exact
32,768-input/128-output requests at concurrency 1 and seed 2026.

The required ordering was enforced: load the original BF16 checkpoint, perform
online INT4 weight quantization during startup with `--quantization wint4`,
mark workers ready, then warm up and time requests. Request E2E therefore
excludes one-time startup quantization. Quantized weights have a theoretical
4x bit-width reduction; actual model or GPU-memory reduction also includes
scales, unquantized parameters, and the BF16 KV cache.

The current FlashAttention baseline succeeded with mean TTFT 7,802.485 ms,
TPOT 14.034 ms, and output throughput 13.345 token/s. InfLLM-v2 activated
sparse prefill during the untimed 32K warmup, then exited with
`CUBLAS_STATUS_INVALID_VALUE` in the LM head; disabling overlap scheduling
produced the same failure. Consequently there is no honest current-source
InfLLM-v2 W4A16 serving speedup. A failed run is neither 0x nor a reason to
substitute an attention-only ratio. See the successful
[FlashAttention result](../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/serving_flash_attn_32k_128_b1.json)
and the [InfLLM-v2 failure record](../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/serving_infllmv2_failure.json).

### Retained successful W4A16 service pair (2026-08-19, historical checkpoint)

The repository also retains a successful paired run that permits a reproducible
serving calculation. Both backends used the same offline-prequantized W4A16
checkpoint, one RTX A6000, two 65,344-input/16-output requests, and concurrency
1. Both input payloads have SHA-256
`1100aec67962ce4cf3cf5e9bc18ab7c1bcc44ff06bbf71d80b058322b4a02aa8`.
This is not a 2026-08-25 current-source result because the current loader no
longer supports that prequantized checkpoint path.

| Serving metric | FlashAttention | InfLLM-v2 | InfLLM-v2 speedup |
| --- | ---: | ---: | ---: |
| Prefill proxy: mean TTFT | 21,628.915 ms | 12,950.952 ms | 1.670x |
| Decode proxy: mean TPOT | 18.143 ms | 35.478 ms | 0.511x |
| Derived mean E2E/request | 21,901.054 ms | 13,483.121 ms | 1.624x |
| Measured total duration | 43.820 s | 26.983 s | 1.624x |
| Output throughput | 0.730 token/s | 1.186 token/s | 1.624x |
| Clean decode throughput | 53.332 token/s | 28.186 token/s | 0.529x |

Here `mean E2E = mean TTFT + (16 - 1) * mean TPOT`. The duration and output
throughput ratios agree because the token totals are identical. Sparse prefill
substantially improves TTFT, while decode is about 1.956x slower. This
short-output workload remains prefill-dominated, yielding about 1.624x overall
E2E speedup. The raw records are
[FlashAttention](../../runs/bench/infllmv2/w4a16_64k_b1_gpu7_20260819/flash_attn_64k_single_prefill.json)
and [InfLLM-v2](../../runs/bench/infllmv2/w4a16_64k_b1_gpu7_20260819/infllmv2_64k_single_prefill.json);
the consolidated calculation is in [summary.json](../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/summary.json).

## Acceptance matrix

| Item | Code/evidence | Status when this document was written |
| --- | --- | --- |
| INF-1: per-request/per-head Stage 1 | `infllmv2_update_compressed_k`, `infllmv2_select_blocks`; backend/operator tests | Current package passed 24/24 focused cases: backend 15/15 and packaged GPU 9/9 |
| INF-2: post-RoPE + paged-cache Stage 2 | `decoder_write_cache_with_rope` before `infllmv2_attention_forward`; ordering and paged-mapping tests | Passed: writer order, already-written cache, cross-page summaries, and paged mapping are covered |
| P0: GQA-tiled Stage 2 | Production-shape CUDA path and NumPy-oracle operator test | Passed; aligned paged K/V movement reduced current 65K/8 Stage 2 to 163.840 us versus 206.848 us locally |
| P2: persistent Stage 2 buffers | Backend workspace reuse and four-output in-place alias tests | Passed; no Stage 2 hot-path `empty`/`full`/`zeros` allocation |
| Stage 1-1: GQA-tiled coarse LSE | Split partial/merge kernels, persistent workspaces, semantic-reference GPU test, Nsight traces | Passed; 14.558x at 32K/1 and 21.151x at 128K/4 |
| Stage 1-2: exact radix TopK | Composite-key CUB selector and 2,048-candidate tie/padding GPU tests | Passed; 2.478x at 32K/1 and 22.545x at 128K/4 |
| Stage 1-3: GQA-tiled block-score | Eight-block K tiles shared across 16 query heads; production GPU oracle tests and Nsight traces | Passed; block-score is 4.382x faster at 32K/1 and 9.391x faster at 128K/4 than the preceding checkpoint |
| P3: calibrated long-context comparison | Current CUDA-event gate, same-shape local-repository rerun, Nsight, and fully restarted 65K/8 service report | Operator gate passed at 1.5458x; current complete service remained at 0.983x dense output throughput, so no universal service-speedup claim is made |
| Implementation documentation | This page, root `rfc.md`, and the current rerun summary | Provided with traceable current-source results; the 2026-08-17 section supersedes earlier performance checkpoints |

## References

- [InfLLM-V2 paper](https://arxiv.org/abs/2509.24663)
- [OpenBMB InfLLM-V2 CUDA implementation](https://github.com/OpenBMB/infllmv2_cuda_impl)
- [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)
- [AngelSlim/Hy3-GGUF](https://huggingface.co/AngelSlim/Hy3-GGUF)
- [llama.cpp PR #25395](https://github.com/ggml-org/llama.cpp/pull/25395)

Hy3-GGUF and llama.cpp PR #25395 are user-specified low-bit/Hy3 engineering references. They are not sources for this InfLLM-V2 selection algorithm or its performance numbers.
