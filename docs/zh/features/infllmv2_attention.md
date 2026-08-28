[English](../../features/infllmv2_attention.md)

# InfLLM-V2 稀疏注意力

FastDeploy 的 `INFLLMV2_ATTN` 后端为 MiniCPM4.1 的 decode 和长上下文 prefill 提供两阶段稀疏注意力：Stage 1 按请求和 KV head 动态选择逻辑 cache 块，Stage 2 通过 `block_tables` 读取已写入的 paged K/V cache。

初始 batch-1 prefill 在 `dense_len` 之前使用 dense FlashAttention，之后每个 128-token query tile 共享一次 Stage 1 选块。每个稀疏 tile 被拆成“完全可见的历史选中块”和“当前一至两个块”：历史分区运行 non-causal FlashAttention，当前分区运行 causal FlashAttention，再用两边精确的 log-sum-exp 合并输出。这样即使 Paddle FlashAttention V2 会把较短 causal query 左上对齐，也能保持正确的因果位置。短请求、mixed batch 和共享 chunked-prefill 路径继续使用 dense prefill，同时仍建立 decode 所需的双尺度语义摘要。

## 实现概览

decode 数据流如下：

```text
raw fused QKV
  -> decoder_write_cache_with_rope
       -> post-RoPE / post-QK-norm Q
       -> 当前 K/V 写入 paged cache
  -> infllmv2_update_compressed_k
  -> infllmv2_select_blocks
  -> infllmv2_attention_forward
       -> block_tables[请求, 逻辑块] -> physical page
       -> causal sparse softmax
```

Stage 1 和 Stage 2 都使用 writer 产生的 post-RoPE query，Stage 2 读取的 K/V 包含当前 decode token。原始 fused QKV 不会被直接送入 sparse attention。

sparse prefill 数据流如下：

```text
raw fused QKV
  -> gqa_rope_write_cache
       -> post-RoPE / post-QK-norm Q/K/V 和 paged K/V cache
  -> infllmv2_update_compressed_k
  -> [0, dense_len) 使用 dense FlashAttention
  -> 每个 128-token query tile 调用一次 infllmv2_select_blocks
  -> gather 选中的 paged K/V
       -> 历史选中页运行 non-causal FlashAttention
       -> 当前一至两个页运行 causal FlashAttention
       -> 使用精确 LSE 加权合并输出
```

默认每次处理 4,096 个 token，即 32 个 query tile，以限制临时 gather K/V 的显存；最后不足 128 token 的 tile 也受支持，prompt 长度不必按 block 对齐。

对每个 query token，请求与 causal 位置由运行时元数据计算：

```text
request = batch_id_per_token[token]
local_offset = token - cu_seqlens_q[request]
position = seq_lens_decoder[request] + local_offset
```

因此 continuous batching 中的不同请求、KV head 和 paged-cache block table 保持隔离；`batch_id_per_token == -1` 的 padding token 不会访问 cache。

### 双尺度 Stage 1

Stage 1 使用两种 K 语义窗口：

| 尺度 | 默认窗口 | 默认步长 | 用途 |
| --- | ---: | ---: | --- |
| fine | 32 | 16 | 为逻辑 KV 块计算精细相关度 |
| coarse | 128 | 64 | 估计每个 query head 的 log-sum-exp 归一化项 |

每个窗口保存 paged K cache 中 K 向量的均值。完整窗口归属于包含其最后一个 token 的 physical page，所以 fine/coarse 窗口可以跨 page。对默认 `block_size=64`，每个 physical block 和 KV head 有 4 个 fine slot 和 1 个 coarse slot。

设 `G(g)` 是 KV head `g` 对应的 GQA query-head 组，`d` 是 head dimension。打分过程为：

```text
LSE[h] = logsumexp_j(dot(Q[h], Kbar_coarse[j]) / sqrt(d))

semantic_score[i, g] = sum(h in G(g)) exp(
    dot(Q[h], Kbar_fine[i]) / sqrt(d) - LSE[h]
)

block_score[b, g] = max(semantic_score[i, g]
                              for fine window i overlapping block b)
```

对 MiniCPM4.1 生产形状（`QH=32`、`KVH=2`、`D=128`、`block_size=64`），coarse 归一化使用参考 OpenBMB CUDA 实现分布式 softmax 结构改写的 GQA-tiled
split kernel。一个 CTA 负责 `(query token, KV head, coarse split)`；16 个 half-warp 将 16 个 Q head 常驻寄存器，协作加载一份对齐的 `float4` K tile（最多16 个 coarse windows），并在整个 GQA 组内复用。每个 split 的 online`(max, sum)` 分布在各 lane，随后由“一 warp/一 query head”的 kernel 做 rescale归并。split partial 是调用方提供的持久化 workspace，因此该路径既没有thread-0 串行窗口循环，也没有逐次调用分配。

生产形状的 fine block-score 路径一次 tile 8 个相邻候选块。一个 CTA 负责 `(query token, KV head, 8-block tile)`；16 个 half-warp 将 GQA query head 常驻寄存器，最多 33 个重叠 fine K row 只通过对齐 `float4` 搬运一次并由整个 GQA 组共享，随后同一组 head contribution 同时为 8 个候选块做 pooling。backend 会跨 decode step 复用 score workspace，因此 kernel 会重写所有 score slot，
包括不足 8 块的 table 尾部。

初始 `init_blocks` 和最近 `window_size` 对应的局部块会被置为正无穷分并参与同一次 top-k，其余候选块按 `block_score` 排序。`local_blocks = window_size / block_size` 且包含当前块，进入稀疏区后的选中预算为 `topk + local_blocks`；因此 `topk` 不是强制块之外的额外预算。默认配置会选中 96 块，其中包含 33 个强制块（1 个初始块和含当前块在内的 32 个局部块）与最多 63 个普通动态块。输出 buffer 为了容纳短上下文选全块，其容量为：

```text
selected_capacity = max(
    topk + window_size / block_size,
    ceil(dense_len / block_size),
)
```

稀疏区选块使用精确的 64-bit 复合 radix key：分数降序、逻辑块 ID 升序。
256-thread CUB block radix sort 按候选规模为每线程分派 1、2、4 或 8 个元素，
单 CTA 可覆盖最多 2,048 个逻辑块（`block_size=64` 时对应 128K token）。它替换
了旧的全候选两两排名，同时保持 tie 时的确定性。短于 `dense_len` 的请求直接
输出所有可见块，不进入排序；超过 2,048 个候选块的模型配置会显式失败。

Stage 1 在 GPU 上输出：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `topk_indices` | `[tokens, kv_heads, selected_capacity]` | 每请求/query token/KV head 的逻辑块 id，未用位为 `-1` |
| `block_scores` | `[tokens, kv_heads, max_blocks_per_seq]` | 动态块分数 |
| `selected_counts` | `[tokens, kv_heads]` | 实际选中块数 |

`topk_indices` 的有效前缀必须是严格递增、无重复且在当前请求可见范围内的逻辑块 ID，其余 slot 必须全为 `-1`。Stage 2 依据第一个 `-1` 判定前缀长度；这是 custom op 的输入前置条件，正常请求中由 Stage 1 保证。runtime 序列长度、`cu_seqlens_q`、token/request 映射及 `block_tables` 页所有权也必须由 FastDeploy 调度器保证自洽；不同活跃请求不得共享可写 physical page。Q/K/V 输入必须为有限值。

MiniCPM4.1 是 GQA 模型，因此元数据按 KV head 生成，不会为同一 GQA 组内的每个 query head 复制一份块列表。Stage 2 使用该 KV head 的块列表服务同组 query heads。

### Paged Stage 2

Stage 2 不假设 logical block 在显存中连续，而是对每个选中块执行：

```text
physical_block = block_tables[request, logical_block]
```

随后仅对该 physical page 中不超过当前请求 `position` 的 token 计算 scaled dot-product attention。`topk_indices` 必须是逻辑块 id，不得传 physical page id。

MiniCPM4.1 的生产形状为 32 个 query heads、2 个 KV heads、`head_dim=128`、`block_size=64`，Stage 2 对此使用 GQA-tiled FlashDecoding 内核。一个 128-thread CTA 负责 `(query token, KV head, KV split)`，一次覆盖共享该 KV head 的全部 16 个 query heads。内核保留 16-token K/V shared-memory tile：warp 0 计算 16x16 QK tensor-core tile，thread 0 至 15 更新各 head 的 online `(max, sum)`，四个 warp 共同计算 probability-times-V tensor-core tiles，并更新 shared FP32 accumulator。完整 16-token K/V tile 从 paged cache 通过对齐的 16-byte `uint4` transaction 搬入 shared memory；因果边界的非完整 tile 显式走带 mask 的标量路径。独立 combine kernel 归并各 split 的 `(accumulator, max, sum)` partial。

每个 KV split 包含两个选中 page，在 batch 1 时仍可提供足够 CTA，同时不会恢复旧实现的 16 倍 K/V 全局显存读取。本地 CUDA 子仓库使用 FlashAttention/CUTE 64-token K/V tile；FastDeploy 保留较小 tile，是因为直接移植 64-token paged tile 需要 47.3 KiB shared memory，并使 128K/batch 4 Stage 2 回退 17.1%。当前 16-token paged tile 约使用 22 KiB，同时保留 GQA 复用、split-KV 与 online softmax。非生产形状走通用正确性路径，不代表 MiniCPM4.1 的性能路径。

每层 backend 持久化持有 Stage 1 元数据、最终 attention 输出以及 FP32 split `(accumulator, max, sum)` tensor；它们都作为 in-place 输入传给 custom op，并在形状不变的 decode step 间复用。Stage 2 CUDA launcher 的热路径不再调用 `paddle::empty`、`paddle::full` 或 `paddle::zeros`。

## 配置与启用

使用环境变量显式选择后端：

```shell
export FD_ATTENTION_BACKEND=INFLLMV2_ATTN
```

参数优先从模型 `config.json` 的 `sparse_config` 读取，其次为模型顶层同名字段，最后使用 MiniCPM4.1 默认值。如需要覆盖，可在模型配置中添加：

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

| 参数 | 默认值 | 约束 |
| --- | ---: | --- |
| `block_size` | 64 | 启动参数 `--block-size`；必须同时被 `kernel_stride` 和 `4 * kernel_stride` 整除 |
| `kernel_size` | 32 | 正整数 |
| `kernel_stride` | 16 | 正整数 |
| `topk` | 64 | 正整数 |
| `dense_len` | 8192 | 至少为 `4 * kernel_size` |
| `init_blocks` | 1 | 非负整数 |
| `window_size` | 2048 | `block_size` 的非负整数倍 |
| `sparse_prefill` | `true` | 是否启用模型专用的初始 prefill 稀疏路径 |
| `prefill_query_chunk_size` | 4096 | 128 的正整数倍 |

`init_blocks` 必须小于 `topk`，以保证初始块和包含当前块的局部集合都能放入输出容量。

32K 长上下文单卡服务示例：

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

## 限制与显存费用

- 只支持 compute capability 8.0 及以上（SM80+）的 NVIDIA CUDA GPU，以及 FP32、FP16 和 BF16。
- paged K/V cache 必须为 rank 4 且未量化；量化 KV cache 会显式失败。
- query heads 数必须被 KV heads 数整除，`head_dim` 必须在 `[1, 256]` 内。
- tensor parallel 不得复制 KV head：`tensor_parallel_size` 必须不大于每层的全局 KV-head 数。当前尚未实现跨副本的 Stage 1 分数归并，因此不支持的配置会显式失败。
- sparse decode 需要同一 paged K cache 先经过 prefill 建立语义摘要。当前 P/D 分离不会传输这些摘要，因此不得在缺失摘要时进入 sparse decode。
- 更换或 reset paged K cache 时必须一起 reset 语义摘要。
- 语义摘要尚未纳入 CUDA Graph 的 cache replacement，因此当前必须关闭 CUDA Graph。
- 推测解码/MTP 当前会显式拒绝，直到每个 proposed token 都能参与 Stage 1 元数据生成和语义 cache 更新。
- sparse prefill 当前仅对 MiniCPM4.1 生产形状（`QH=32`、`KVH=2`、`D=128`、`block_size=64`）、FP16/BF16 cache、初始 causal batch-1 请求启用；其他布局回退到 dense prefill。
- 本次 32K 验收使用单个调度器级 32K prefill 请求（内部仍按 4,096 token 的 sparse batch 处理），因此没有覆盖调度器级多 chunk prefill。使用 `--max-num-batched-tokens 8192 --enable-chunked-prefill` 时，现有 FlashAttention 和 InfLLM-V2 服务都会在第三个 8K prefill chunk 触发 Paddle `optional<T>::get()` 未初始化断言；这说明问题位于共享的 chunked-prefill 路径，而不是 sparse Stage 1/Stage 2。本实现没有修改该公共路径，32K 对照统一使用 `--max-num-batched-tokens 32768`。

每层、每个 physical block 的理论额外语义-cache 字节数为：

```text
dtype_bytes * kv_heads * head_dim * (
    block_size / kernel_stride
    + block_size / (4 * kernel_stride)
)
```

对 MiniCPM4.1 的 BF16、`kv_heads=2`、`head_dim=128` 和默认窗口，该理论值为 2,560 bytes/block/layer。这是由形状推导的静态费用，不是 INF-3 实测峰值显存。

## 构建与回归

InfLLM-V2 修改了 CUDA custom ops，运行算子测试前需要重新构建。下列示例使用项目 `.venv`；请按实际 CUDA 安装路径和 GPU compute capability 调整 `CUDA_HOME` 与架构列表：

```shell
export CUDA_HOME=/path/to/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PWD}"

MAX_JOBS=32 FD_BUILD_RESUME=1 \
  bash build.sh 0 "${PWD}/.venv/bin/python" false "[86]"
```

与本特性直接相关的回归命令：

```shell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/model_executor/test_infllmv2_attention_backend.py \
  tests/operators/test_infllmv2_attention_forward.py
```

高层测试覆盖 writer/Stage 1/Stage 2 顺序、post-RoPE query 和语义 cache 生命周期；GPU 算子测试覆盖跨 page 语义窗口、每请求/每 KV head 选块、paged mapping、短上下文闭环与非法元数据。该特性不要求文档测试，也不应修改 `tests/model_executor/test_minicpm41.py`。

## Dense/Sparse 长上下文对照

`benchmarks/benchmark_infllmv2.py` 有七个子命令：

- `selector`：检查 raw selector trace 并生成未绑定负载的命中率摘要；该摘要仅供查看，不能附加到 sparse 计时结果。
- `sparse-diagnostic`：对启用 trace 的 sparse 服务执行一次诊断负载，自动向 `benchmark_serving.py` 传入 `--no-warmup`，并在 selector diagnostic JSON 中将新生成的 trace 与精确 prompt-token 负载绑定。
- `run`：调用 `benchmark_serving.py` 收集吞吐、TTFT 和 TPOT，同时采样指定 GPU 的显存；sparse 计时运行必须通过 `--selector-diagnostic` 提供已绑定的诊断 JSON。
- `report`：仅在 dense/sparse workload 指纹完全一致且两次运行的 physical GPU UUID 相同时生成对照 JSON 和 Markdown 表格。
- `operators`：使用 CUDA event 分别测量 compressed-K update、Stage 1、Stage 2 和完整 sparse decode 算子链；默认矩阵为 32K/并发 1 与 128K/并发 4。
- `prefill`：使用 CUDA event 对比完整的 Paddle dense/sparse prefill 路径。
- `cuda-impl`：对本地 `infllm_v2` PyTorch extension checkout 执行等价 decode 负载。

逐算子矩阵应与服务计时分开运行：

```shell
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_infllmv2.py operators \
  --gpu-index 7 \
  --scenario 32768:1 \
  --scenario 131072:4 \
  --output runs/bench/infllmv2/operators.json
```

JSON 会为每个场景记录 CUDA mean/median/P10/P90/P99、host wall time、源码哈希、选中 token 比例和持久化 workspace 字节数，并校验 Stage 2 的四个输出均与调用方缓冲区共享存储。该合成 benchmark 用于隔离 kernel 成本，不能替代 TTFT/TPOT 服务对照。

sparse prefill 使用独立的完整链路 benchmark：

```shell
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. .venv/bin/python \
  benchmarks/benchmark_infllmv2.py prefill \
  --device gpu:0 \
  --context-length 16384 \
  --context-length 32768 \
  --output runs/bench/infllmv2/sparse_prefill.json
```

sparse 计时包含 compressed-K 更新、dense prefix、Stage 1、paged-cache gather、两个 FlashAttention 分区及 LSE 合并。2026-08-18 在 NVIDIA RTX A6000、BF16、batch 1 上，32K 实测 dense 90.300 ms、sparse 46.005 ms，即 **1.963x**；16K 为 21.256 ms 对 18.654 ms，即 1.139x，说明该优化面向长上下文。可追溯结果位于 `runs/bench/infllmv2/sparse_prefill_20260818.json`。

### 32K/batch 1 理论收益上限

默认配置下，sparse Stage 2 选择 96 个、每个 64 token 的 page。在 32K 上实际参与 Stage 2 的是 6,144 / 32,768 个可见 token，即 18.75%。即便假设完全受显存带宽限制且没有任何固定开销，Stage 2 的 token 缩减理论上限也只有 `32768 / 6144 = 5.333x`。端到端 decode 不可能超过该值，并受到更严格的 Amdahl 上限约束：

```text
speedup <= 1 / ((1 - f_stage2) + f_stage2 / 5.333)
```

其中 `f_stage2` 是 dense decode 中 Stage 2 attention 的耗时占比。Stage 1、compressed-K update、split merge、launch 开销、MLP/MoE 以及 batch 1 的 GPU 欠占用均不属于理想 token 缩减，都会压低实测收益。128K 的选中 token 比例更低，但 Stage 1 需要扫描更多语义窗口，因此不能把 token 比例直接当作实测加速比。

### 选块命中率

命中率是所有“请求/query token/KV head”样本的 dense-reference block recall 微平均：

```text
sum(|selected_blocks ∩ reference_blocks|) / sum(|reference_blocks|)
```

`reference_blocks` 应使用同一 post-RoPE query 和同一 paged K cache 的精确 dense attention 生成，并按 GQA 组聚合到 KV head 粒度。raw sample 格式为：

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

数组中的 block id 必须非负且不重复，`reference_blocks` 不得为空。该样例只说明 JSON 格式，其数字不是模型实测结果。

后端可以在独立诊断运行中直接生成这些样本。请先设置 trace 环境变量，再启动专用的 `INFLLMV2_ATTN` 诊断服务：

```shell
export FD_INFLLMV2_SELECTOR_TRACE_PATH="${PWD}/runs/bench/infllmv2/selector_samples.json"
export FD_INFLLMV2_SELECTOR_TRACE_RANK=0
export FD_INFLLMV2_SELECTOR_TRACE_LAYER=0
export FD_INFLLMV2_SELECTOR_TRACE_MAX_SAMPLES=16
```

启用 trace 时必须关闭 CUDA Graph。后端在 Stage 1 之后使用同一 post-RoPE query、`block_tables` 和已写 paged K cache 计算 FP32 dense softmax；对共享一个 KV head 的 query heads 汇总每个逻辑块的 attention mass，再取与 `selected_count` 相同数量的最高质量块作为 reference。trace 会产生 GPU 同步、device-to-host 复制和 dense attention 额外计算，因此诊断延迟不得作为性能结果。诊断前 trace 路径必须不存在，服务也会显式拒绝覆盖已有文件。

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

`sparse-diagnostic` 会通过传入 `--no-warmup` 自动关闭常规 warm-up 请求，不会生成需要上报的 warm-up 结果。它会把实际生成的 prompt-token 摘要、workload 指纹和 trace 来源一起记录；之后的 sparse 计时 `run` 会拒绝负载不匹配的诊断文件。

也可以单独汇总 raw trace 便于查看，但该未绑定摘要不能被 `run` 接受：

```shell
.venv/bin/python benchmarks/benchmark_infllmv2.py selector \
  --samples runs/bench/infllmv2/selector_samples.json \
  --output runs/bench/infllmv2/selector_summary.json
```

### 固定负载运行

诊断完成后必须完全停止启用 trace 的 sparse 服务。raw trace 文件需要保留，因为绑定后的诊断会按路径和 SHA256 再次校验它；但任何计时前都必须禁用 trace 并重新启动服务：

```shell
unset FD_INFLLMV2_SELECTOR_TRACE_PATH
```

诊断服务和计时服务必须是不同进程。请确认计时用 `INFLLMV2_ATTN` 服务启动时没有 trace-path 环境变量，并在整个计时期间保持 trace 关闭。

正式计时对照中，先使用 `FD_ATTENTION_BACKEND=FLASH_ATTN` 启动 dense 服务，运行：

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

停止 dense 服务，在同一块空闲 GPU 上使用 `FD_ATTENTION_BACKEND=INFLLMV2_ATTN` 和完全相同的服务参数启动未启用 trace 的 sparse 服务，再运行：

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

`sparse-diagnostic`、dense `run` 和 sparse `run` 的负载参数必须完全一致，包括 model、tokenizer、endpoint、seed、请求数、输入/输出长度、request rate 和并发。脚本会校验实际生成的 prompt tokens，并拒绝不匹配的负载指纹。

`--gpu-index` 是 `nvidia-smi --id` 使用的 physical GPU 索引。如果设置 `CUDA_VISIBLE_DEVICES=7`，这里应传 `--gpu-index 7`，而不是进程内的逻辑索引 0。测试时应保证该 GPU 没有其他负载，dense/sparse 之间完全重启服务。

每个 `run` 输出中的 `gpu_memory.gpu_uuid` 标识 physical device，`baseline_mib` 是请求开始前已启动服务的显存基线，`peak_mib` 是 GPU 总已用显存峰值，`peak_delta_mib` 是两者差值。`report` 会拒绝 GPU UUID 不同的 dense/sparse 输入，并在结果中保留 UUID、baseline、peak 和 peak delta；验收报告必须完整保留这些数据。

### 保留的 P0/P2 检查点结果

最终 CUDA event 矩阵保存在 `runs/bench/infllmv2/tmp/p3_operator_matrix.json`。它使用 MiniCPM4.1 生产形状（`QH=32`、`KVH=2`、`D=128`、BF16），在 NVIDIA RTX A6000 上预热 50 次、测量 200 次。下表均为设备侧中位数。

| 场景 | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | 完整 sparse 算子链 (us) | Stage 2 持久化字节数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 49.152 | 191.488 | 111.616 | 380.928 | 1,073,152 |
| 128K / 并发 4 | 49.152 | 1,662.976 | 215.040 | 1,939.968 | 4,292,608 |

P0/P2 修改前的 Stage 2 基线保存在 `runs/bench/infllmv2/tmp/p0_p2_operator_baseline.json`。同一 GPU、相同形状的结果为：

| 场景 | P0 前 Stage 2 (us) | GQA-tiled Stage 2 (us) | 加速比 | 耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 135.168 | 111.616 | 1.211x | 17.42% |
| 128K / 并发 4 | 367.616 | 215.040 | 1.710x | 41.50% |

P0 独立检查点的 Stage 2 实测为 112.144 us 和 215.040 us，相对保留基线分别为 1.205x 和 1.710x；32K 与最终 111.616 us 的差异属于运行波动。

矩阵同时验证最终输出和三个 FP32 partial tensor 均与调用方传入的存储 alias。这里的字节数是每个 layer-backend 实例的开销：并发 1 时 Stage 2 持久化 1,073,152 bytes，并发 4 时为 4,292,608 bytes；每个实例只在形状变化时分配，不再每次 layer 调用分配。对这个 32 层 checkpoint，对应总量为 32.75 MiB 和 131.00 MiB。P2 检查点的 32K/1 Stage 2 为 112.640 us，P0 检查点为 112.144 us（+0.44%）；完整算子链从 381.952 us 变为 380.928 us（-0.27%），128K/4 Stage 2 均为 215.040 us。该变化处于运行噪声范围，说明 in-place 持久化没有引入热路径回归。P0 与 P2 分步结果分别保存在同一临时目录的 `p0_gqa_tiled_operator.json` 和 `p2_persistent_workspace_operator.json`。

本次 checkpoint 的 `max_position_embeddings` 与 `rope_scaling.original_max_position_embeddings` 都是 65,536；128K 服务请求会违反模型契约。因此验收要求的 128K/并发大于 1 对照明确采用合成逐算子测试，不把它表述为端到端模型结果。

### Stage 1 选块优化

后续 Stage 1 GQA-tiled coarse LSE 与精确 radix TopK 的分步结果分别保存在
`runs/bench/infllmv2/tmp/stage1_p1_gqa_tiled_result.json` 和
`runs/bench/infllmv2/tmp/stage1_p2_radix_topk_result.json`，合并后的 CUDA event
结果为 `stage1_p1_p2_optimized.json`；同目录的 Nsight Systems trace 与 SQLite
导出提供逐 kernel 归因。测试使用同一块 RTX A6000 和 BF16 生产形状，CUDA
event 数据预热 20 次、测量 100 次，表中为中位数。

| 场景 | 优化前 Stage 1 (us) | 优化后 Stage 1 (us) | Stage 1 加速 | 优化前算子链 (us) | 优化后算子链 (us) | 算子链加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 191.488 | 82.944 | 2.309x | 381.952 | 246.784 | 1.548x |
| 128K / 并发 4 | 1,659.904 | 820.224 | 2.024x | 1,935.360 | 1,052.672 | 1.839x |

| 场景 | 优化前 coarse LSE (us) | GQA-tiled coarse LSE (us) | 加速 | 优化前 TopK (us) | Radix TopK (us) | 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 111.155 | 7.635 | 14.558x | 34.643 | 13.978 | 2.478x |
| 128K / 并发 4 | 479.877 | 22.688 | 21.151x | 509.055 | 22.579 | 22.545x |

在该检查点的 128K/并发 4 下，block-score 是 Stage 1 的主要 kernel（773.271 us，
约占 Stage 1 的 94%），因此成为后续选块瓶颈。该检查点的隔离 extension 通过
10 个 SM86 GPU 正确性用例，包括生产形状 BF16 语义参考以及 2,048 候选的精确
tie/padding 用例；backend 回归通过 39 个用例。该检查点完成全量 package
重编译后，packaged SM86 算子回归 15 个用例全部通过。下一节给出 block-score
瓶颈的最终处理结果。

### GQA-tiled block-score 与 Stage 2 最终优化

后续实现参考了 OpenBMB 官方仓库中的 split-KV 与分布式 softmax 结构；参考仓库
下载在 `runs/bench/infllmv2/tmp/infllmv2_cuda_impl`，固定 commit 为
`93cf2ec28e5a7acebe3f0bb7329b6c73a1be91f6`。FastDeploy 仍保留自己的 paged
cache、持久化 workspace 和 in-place custom-op 契约。

最终隔离 CUDA-event 矩阵为
`runs/bench/infllmv2/tmp/stage1_p3_gqa_blockscore/operator_optimized_halfwarp.json`，
使用同一块 RTX A6000、BF16 生产形状，预热 20 次并测量 100 次。

| 场景 | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | 完整 sparse 算子链 (us) |
| --- | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 49.152 | 48.128 | 59.392 | 149.504 |
| 128K / 并发 4 | 49.152 | 131.072 | 124.928 | 317.440 |

相对保留的 P0/P2 矩阵，32K/1 完整算子链从 380.928 降至 149.504 us
（2.548x，耗时下降 60.75%），128K/4 从 1,939.968 降至 317.440 us
（6.111x，耗时下降 83.64%）。相对紧邻的 coarse-LSE/radix-TopK 检查点，Stage 1
在 32K/1 与 128K/4 分别加速 1.723x 和 6.258x，Stage 2 分别加速 1.879x 和
1.787x。

同一临时目录保存了 Nsight Systems report、SQLite 导出和生成的 kernel-summary
CSV。下表按 kernel 平均耗时比较上一检查点与最终实现。

| Kernel | 32K 优化前/最终 (us) | 加速 | 128K/4 优化前/最终 (us) | 加速 |
| --- | ---: | ---: | ---: | ---: |
| Fine block-score | 58.585 / 13.370 | 4.382x | 773.271 / 82.341 | 9.391x |
| Stage 2 partial | 70.386 / 40.153 | 1.753x | 152.721 / 95.404 | 1.601x |
| Stage 2 merge | 37.548 / 15.059 | 2.493x | 72.025 / 25.997 | 2.770x |

当前源码的隔离 extension 已通过 14/14 个 GPU 用例，覆盖 FP32/FP16/BF16 paged
Stage 2、生产 BF16 GQA Stage 1/2、精确 2,048-way tie、复用 buffer 尾部和非法
契约。完整重编译后的生产 package 与配对服务结果在下文继续报告。

在模型支持的 32K 服务负载上，保留的 P0/P2 检查点报告
`runs/bench/infllmv2/tmp/p3_report_32k_current.json` 对比了从当时同一 packaged
build 启动、服务参数完全相同的 dense 与 sparse 服务；它早于上述 Stage 1 选块
优化。两份输入的 workload ID 均为
`b0fd4419f876adfa40e4d92a6e46d56c95c0f5f37e0539abd1a32c8208f6c078`，
prompt-token hash 与 physical GPU UUID 相同，且已通过 `report` 命令校验。

| 指标 | Dense | P0/P2 检查点 sparse | Sparse / Dense |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 64.4531%（990 / 1536） | — |
| Request throughput (req/s) | 0.092799 | 0.084076 | 0.906x |
| Output throughput (token/s) | 11.878301 | 10.761664 | 0.906x |
| Total-token throughput (token/s) | 3052.723423 | 2765.747673 | 0.906x |
| Mean / median / P99 TTFT (ms) | 7368.456 / 7362.350 / 7399.758 | 7404.749 / 7389.901 / 7467.811 | 1.005x / 1.004x / 1.009x |
| Mean / median / P99 TPOT (ms) | 26.742 / 26.744 / 26.794 | 35.252 / 35.408 / 35.514 | 1.318x / 1.324x / 1.325x |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 0.996x / 0.997x / 1.000x |

服务负载为 4 个确定性 random-token 请求，input 32768、output 128、并发 1、request rate `inf`、seed 2026。服务使用 TP1、BF16、`block_size=64`、默认 sparse 配置 `32/16/64/8192/1/2048`，关闭 prefix cache 和 CUDA Graph，并使用单个 32K prefill chunk。测试环境为 NVIDIA RTX A6000 48 GiB（UUID `GPU-56bb1092-c218-9cd3-ad85-9a2735522d28`）、CUDA toolkit 12.8、Paddle 3.3.1（编译 CUDA 12.6）和 Python 3.12.13。

该保留服务报告不是新 Stage 1 kernel 的端到端测量：其中 sparse 输出吞吐为
dense 的 0.906x，mean TPOT 为 dense 的 1.318x。当前源码的算子链在 32K/1
缩短 35.39%，在 128K/4 缩短 45.61%。在该检查点，仍需完成 packaged server
重编译和同负载服务复测；已完成的复测见下节。更早的
`runs/bench/infllmv2/report.json` 仍作为历史结果保留（输出吞吐为 dense 的
0.311x、mean TPOT 为 dense 的 7.945x），不得将它当成当前 kernel 的结果。

### 历史 Stage 1 优化 packaged 服务检查点

该检查点使用当时的源码完整重编译了 `fastdeploy_ops` package。安装库当时的 SHA-256 为
`cb37a51292ed2f92139a199f748a1e8fb014d6c7774284cfdc6bfe535ed2d5e4`，
packaged GPU 算子回归 15/15 通过。构建、正确性、服务与 benchmark 日志均保存在
`runs/bench/infllmv2/tmp/retest_stage1_20260813/`。

通过一致性校验的报告为
`runs/bench/infllmv2/tmp/retest_stage1_20260813/report_32k.json`。dense 与 sparse
使用相同 workload ID、上述相同 prompt-token hash、相同 physical GPU UUID，且两种
服务之间完全重启。正式计时的 sparse 服务未开启 selector trace。

| 指标 | Dense | Stage 1 检查点 sparse | Sparse / Dense |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 59.2448%（910 / 1536） | — |
| Request throughput (req/s) | 0.092539 | 0.087938 | 0.950x |
| Output throughput (token/s) | 11.845043 | 11.256072 | 0.950x |
| Total-token throughput (token/s) | 3044.176118 | 2892.810553 | 0.950x |
| Mean / median / P99 TTFT (ms) | 7403.585 / 7401.545 / 7445.184 | 7414.495 / 7413.212 / 7430.486 | 1.001x / 1.002x / 0.998x |
| Mean / median / P99 TPOT (ms) | 26.710 / 26.717 / 26.747 | 31.071 / 31.058 / 31.133 | 1.163x / 1.162x / 1.164x |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 0.996x / 0.997x / 1.000x |

相对保留的 P0/P2 sparse 检查点，mean sparse TPOT 从 35.252 ms 降到
31.071 ms（下降 11.86%，加速 1.135x），sparse 输出吞吐从 10.761664 提升到
11.256072 token/s（+4.59%）。重复测量的 dense mean TPOT 仅变化 -0.12%，因此
sparse 收益不能由更快的 dense 基线解释。Selector 诊断会跟随实际生成的 decode
轨迹；两次诊断具有相同 prompt 指纹，但不保证生成 token 相同，因此不能把本次
59.24% 与历史 64.45% 直接当成受控的 selector 质量变化。

在这个历史检查点，sparse 虽有改善，但在 32K/batch 1 尚未反超 dense：mean TPOT 仍高
4.361 ms（dense 的 1.163x），输出吞吐为 dense 的 0.950x。5.333x 的 token
缩减上限只适用于 Stage 2 K/V 计算。Dense FlashAttention 本身已高度融合；sparse
decode 还需支付 compressed-K update、block-score/coarse-LSE/TopK 选块、split
merge、32 层 kernel launch，以及相同的 MLP/MoE 成本，batch 1 又无法充分占满
GPU。本检查点中这些固定成本仍超过 Stage 2 节省的 K/V 读取，因此“稀疏”本身不
保证端到端更快。

### 最终 GQA-tiled package 验证与服务复测

完成 block-score 和 Stage 2 最终优化后，再次完整重编译了 SM86 生产 package。
仓库内
`fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` 的
SHA-256 为
`368174499c47eb991595616988f1c2a65bd6384f8cf8353d366511784883bb94`。
backend 回归 39/39 通过，packaged GPU operator 15/15 通过；成功构建与测试日志
保存在 `runs/bench/infllmv2/tmp/stage1_p3_gqa_blockscore/full_build/`。

最终 package 的 CUDA-event 结果为
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/operators_packaged.json`。
该文件记录最终源码 hash，验证 Stage 2 四个输出 alias，并给出以下设备侧中位数：

| 场景 | Compressed-K update (us) | Stage 1 (us) | Stage 2 (us) | 完整 sparse 算子链 (us) |
| --- | ---: | ---: | ---: | ---: |
| 32K / 并发 1 | 49.152 | 37.888 | 59.392 | 149.504 |
| 128K / 并发 4 | 49.152 | 131.072 | 125.952 | 316.416 |

相对保留的 P0/P2 算子链，32K/1 和 128K/4 分别加速 2.548x 与 6.131x；
相对 P0 前的 Stage 2 基线，最终 packaged Stage 2 分别加速 2.276x 与 2.919x。
上文隔离结果的 128K Stage 2/算子链为 124.928/317.440 us，package 复测为
125.952/316.416 us，差异属于运行波动。

最终配对服务结果位于
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/`。它继续使用相同 workload
ID、prompt-token hash、physical GPU UUID 和服务参数。sparse 服务独立重启并重复
测量两次；两个正式计时服务都没有开启 selector trace。

| 指标 | Dense | 最终 sparse 第 1 次 | 最终 sparse 重复测量 |
| --- | ---: | ---: | ---: |
| Selector block hit rate | — | 57.5521%（884 / 1536） | 使用同一绑定诊断 |
| Request throughput (req/s) | 0.092431 | 0.073803（0.798x） | 0.074985（0.811x） |
| Output throughput (token/s) | 11.831154 | 9.446807（0.798x） | 9.598114（0.811x） |
| Total-token throughput (token/s) | 3040.606665 | 2427.829371（0.798x） | 2466.715343（0.811x） |
| Mean / median / P99 TTFT (ms) | 7394.920 / 7395.463 / 7436.516 | 11214.213 / 11260.205 / 11902.204 | 10500.386 / 10554.044 / 10999.020 |
| Mean / median / P99 TPOT (ms) | 26.876 / 26.880 / 27.012 | 18.315 / 19.624 / 25.340 | 22.248 / 22.148 / 24.921 |
| Baseline / peak / peak-delta GPU memory (MiB) | 19493 / 24421 / 4928 | 19417 / 24345 / 4928 | 19417 / 24345 / 4928 |

两次 sparse 的客户端观测 mean TPOT 均低于 dense，分别下降 31.85%（1.467x）和
17.22%（1.208x）；这是本实现首次在同负载 32K 对照中得到 sparse TPOT 低于 dense
TPOT。但它不是所有端到端指标都加速：sparse TTFT 高 42.0%-51.6%，128-token
短输出的 output throughput 仍只有 dense 的 0.798x-0.811x。

最终 sparse 流式输出还存在 burst：两次分别有 45.08% 和 37.20% 的 ITL 小于
1 ms，而 dense 为 0%。排除 burst 后，benchmark 给出的 clean decode rate 为
dense 37.21 token/s、sparse 30.04/28.26 token/s。因此这里把较低 TPOT 表述为可
重复的客户端观测延迟并报告区间，不把单次最佳 1.467x 重新解释成稳定的 kernel
吞吐；上文 CUDA-event 与 Nsight 数据才是 kernel 级证据。

这也解释了为什么 sparse 理论上有优势，却不保证每个指标都更快：32K 下 Stage 2
只读取 dense token 的 18.75%，但 selector、merge、launch、prefill/TTFT 以及非
attention 模型计算仍然存在。完整原始结果与解释汇总在
`runs/bench/infllmv2/tmp/final_gqa_decode_20260813/summary.md`。

### CUDA 恢复后的基线复测（2026-08-16）

上面的 checkpoint 章节保留为优化历史。该检查点的当前源码基线结果是
`runs/bench/infllmv2/service_gate_65k8_20260816/summary.md`；该结果来自重新完整
构建的 SM86 package，没有沿用历史结论。build 产物与仓库实际加载库逐字节一致，
大小为 1,040,001,400 bytes，SHA-256 为
`05e5b67e43d0da09bcbb884ae5b78fba23b8a3a99e31be69a3216c7a5d00e2c7`。
当前 package 的聚焦回归 24/24 通过，覆盖跨 page compressed-K、Stage 1
score/Top-K、paged Stage 2，以及独立 NumPy/Paddle reference。

在模型契约内的 65,344-token、batch 8 BF16 负载上（`QH=32`、`KVH=2`、
`D=128`、选中 96 pages），CUDA-event 中位数为：

| 实现 | Stage 1 + Top-K (us) | Stage 2 (us) | Sparse 算子链 (us) | Dense 基线 (us) |
| --- | ---: | ---: | ---: | ---: |
| FastDeploy | 349.184 | 318.464 | 650.240 | 785.408 |
| 本地 `infllmv2_cuda_impl` | 318.464 | 208.896 | 573.440 | 1,563.648 |

FastDeploy 相对自身 dense 的算子门禁为 1.2079x。相对本地子仓库，FastDeploy 的
Stage 1 + Top-K、Stage 2 和稀疏链分别慢 1.0965x、1.5245x 和 1.1339x。
两个实现的 dense reference 不同，因此 dense 列不得跨实现直接比较。page 对齐的
合成算子有效率为 98,304 / 1,045,504 = 9.40255%。绑定服务 trace 中当前 causal
page 只有部分 token 可见，实际参与率为 97,332 / 1,045,556 = 9.30911%，整页预设
为 9.40208%，差 -0.09296 个百分点。

Nsight 显示 Stage 2 main 是最大稀疏 kernel（286.205 us），fine block-score
次之（249.816 us）。本轮只尝试了一个 64-token shared-page Stage 2 结构：它虽然
改善 32K/batch 1，却使 128K/batch 4 Stage 2 回退 17.1%；Nsight 将回退定位到
47.3 KiB shared-memory tile 导致的 occupancy 降低，因此该实验已撤回。当前保留
16-token tile，并继续使用 GQA K/V 共享、split-KV 和 online softmax。

随后分别完整重启 dense 与 sparse 服务，以相同的 65,344 输入、128 输出、8 请求、
并发 8 负载计时。通过校验的 workload ID 为
`4068415f92b013c216341f8461e29f42c979aa0aa76070b9677844817d68a1fd`。
完整服务没有超过 dense：输出吞吐 5.439 对 5.531 token/s（0.983x），mean TTFT
108,132 对 105,124 ms（1.029x），mean TPOT 628.660 对 627.799 ms（1.001x）。
Selector micro-recall 为 53.84%（827/1,536）。

### Paged K/V 向量化权威复测（2026-08-17）

当前源码的权威证据是
`runs/bench/infllmv2/service_gate_vector_load_20260817/summary.md`。SM86 package
再次完整重建，build 产物与仓库实际加载库逐字节一致，大小为 1,040,001,400 bytes，
SHA-256 为
`1370f08391426352349d420b9268a769d8960615d32a717f1bb66a349dca59dc`；
源码 header SHA-256 为
`883dbc1f5bfbfa8748bb83b3436766f7681d034a26de2e1811661aa738ccd6ae`。
聚焦回归再次 24/24 通过。

本轮只改变生产 Stage 2 完整 tile 的 paged K/V 搬运：global-to-shared 标量循环改为
对齐的 16-byte `uint4` load/store，因果边界的非完整 tile 仍显式走标量 mask 路径。
Stage 1、CUB radix Top-K、16-token 数据布局、四 warp tensor-core 并行结构、每 split
两个 pages、online softmax 与 split combine 均未改变。与本地子仓库 64-token
FlashAttention/CUTE tile 的差异仅保留在已有 paged-cache occupancy 证据要求之处，
没有改写为更简单的标量 attention。

同一 65,344-token、batch 8 BF16 负载，20 次 warmup 加 100 次 CUDA events 的中位数为：

| 实现 | Stage 1 + Top-K (us) | Stage 2 (us) | Sparse 算子链 (us) | Dense 基线 (us) |
| --- | ---: | ---: | ---: | ---: |
| FastDeploy | 329.728 | 163.840 | 507.424 | 784.384 |
| 本地 `infllmv2_cuda_impl` | 304.112 | 206.848 | 547.840 | 1,553.408 |

相对 2026-08-16 FastDeploy 基线，Stage 2 提升 1.9438x，稀疏链降低 21.96%；
FastDeploy 相对自身 dense 的门禁提高到 1.5458x。相对本轮重新运行的本地实现，
FastDeploy Stage 2 快 1.2625x、稀疏链快 1.0796x，Stage 1 + Top-K 仍慢
1.0842x。两个框架的 dense reference 不同，仍不得跨实现比较 dense 列。

Nsight Systems 测得 Stage 2 main 为 145.918 us，低于旧值 286.205 us；fine
block-score 现为最大稀疏 kernel（255.564 us），所以下轮应转向 Stage 1，而不是继续
调整 Stage 2 block 数。合成算子有效参与率为 98,304 / 1,045,504 = 9.40255%；
当前绑定服务 trace 为 97,332 / 1,045,556 = 9.30911%，整页预设为 9.40208%，
相差 -0.09296 个百分点。

随后完整重启 dense 与无 trace sparse 服务，并验证相同 workload ID。完整服务仍未
超过 dense：sparse 输出吞吐 4.832 对 4.915 token/s（0.983x），mean TTFT
133,586 对 131,160 ms（1.018x），mean TPOT 614.727 对 605.495 ms（1.015x）。
Selector micro-recall 为 53.39%（820/1,536）。算子收益成立，但不能据此声称该长
prefill、128-token 输出负载具有通用端到端加速。

## W4A16 attention 与服务端到端记录（2026-08-25）

本节采用统一约定：延迟加速比 = `FlashAttention 延迟 / InfLLM-v2 延迟`，吞吐加速比 =
`InfLLM-v2 吞吐 / FlashAttention 吞吐`；因此大于 1 才表示 InfLLM-v2 更快。attention
算子与完整服务必须分开看：前者只计 attention kernel/算子链，后者还包含 W4A16
Linear、MLP、归一化、采样、调度与 API 开销。

### 当前源码的独立 attention 实测

环境为独占 RTX A6000（SM86），MiniCPM4.1 生产形状 `QH=32`、`KVH=2`、
`D=128`、`block_size=64`、batch 1、BF16 Q/K/V 与 BF16 KV cache。Prefill 使用 20
次 warmup 加 100 次 CUDA-event 计时，decode 使用 50 次 warmup 加 200 次计时。

| 阶段 | 上下文 | FlashAttention 中位延迟 | InfLLM-v2 完整链中位延迟 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| Prefill attention | 16,384 | 20.824 ms | 18.514 ms | 1.125x |
| Prefill attention | 32,768 | 88.638 ms | 45.810 ms | 1.935x |
| Decode attention | 32,768 | 62.464 us | 181.248 us | 0.345x |

32K decode 的完整链由 update 16.384 us、Stage 1 95.232 us 和 Stage 2 58.368 us
组成。Stage 2 单独相对 dense 为 1.070x，但完整链只有 0.345x，即实际慢 2.902x；
不得用 Stage 2 单项代替完整 decode attention 加速比。原始记录见
[prefill JSON](../../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/attention_prefill_16k_32k_b1.json)
与 [decode JSON](../../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/attention_decode_32k_b1.json)。

### W4A16 服务口径与当前状态

W4A16 表示 Linear 权重为 INT4、激活为 BF16；attention Q/K/V 和 KV cache 仍是
BF16。仅比较 attention backend，其他服务参数固定为 TP1、`max_model_len=65536`、
`block_size=64`、`max_num_seqs=1`、`max_num_batched_tokens=32768`、关闭 prefix
cache、chunked prefill 与 CUDA Graph。负载为每请求 32,768 输入、128 输出，4 个请求，
并发 1，seed 2026。

量化顺序严格为：读取原始 BF16 checkpoint -> 服务启动期执行 `--quantization wint4`
在线 INT4 权重量化 -> worker 就绪 -> 才开始 warmup 和计时。因此请求 E2E 不含一次性
启动量化时间；W4A16 对被量化权重的理论位宽压缩是 4x，实际模型/显存压缩还需计入
scale、未量化参数与 BF16 KV cache。

FlashAttention 当前基线成功：平均 TTFT 7,802.485 ms、TPOT 14.034 ms、输出吞吐
13.345 token/s。InfLLM-v2 在计时前的 32K warmup 已激活 sparse prefill，随后在
LM head 报 `CUBLAS_STATUS_INVALID_VALUE` 并退出；关闭 overlap schedule 后相同。因此
当前源码没有可报告的 InfLLM-v2 W4A16 服务加速比，不能将失败样本记为 0x，也不能用
attention 算子比值冒充 E2E。证据见 [FlashAttention 原始结果](../../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/serving_flash_attn_32k_128_b1.json)
和 [InfLLM-v2 失败记录](../../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/serving_infllmv2_failure.json)。

### 已保存的 W4A16 成功服务对照（2026-08-19，历史 checkpoint）

为保留可计算的服务数据，下面单独列出仓库内已成功完成的配对记录。两端使用同一个
预先离线量化完成的 W4A16 checkpoint、同一 RTX A6000、每请求 65,344 输入和 16
输出、2 请求、并发 1；两份输入 payload SHA-256 都是
`1100aec67962ce4cf3cf5e9bc18ab7c1bcc44ff06bbf71d80b058322b4a02aa8`。该结果不代表
2026-08-25 当前源码，因为当前代码已不再支持这个预量化 checkpoint 的加载路径。

| 服务指标 | FlashAttention | InfLLM-v2 | InfLLM-v2 加速比 |
| --- | ---: | ---: | ---: |
| Prefill 代理：mean TTFT | 21,628.915 ms | 12,950.952 ms | 1.670x |
| Decode 代理：mean TPOT | 18.143 ms | 35.478 ms | 0.511x |
| 推导 mean E2E/request | 21,901.054 ms | 13,483.121 ms | 1.624x |
| 实测总时长 | 43.820 s | 26.983 s | 1.624x |
| 输出吞吐 | 0.730 token/s | 1.186 token/s | 1.624x |
| clean decode 吞吐 | 53.332 token/s | 28.186 token/s | 0.529x |

其中 `mean E2E = mean TTFT + (16 - 1) * mean TPOT`。相同 token 总数下，总时长比与
输出吞吐比一致。结果说明 sparse prefill 带来明显 TTFT 收益，但 decode 约慢 1.956x；
本负载输出很短，prefill 收益仍使整体 E2E 达到约 1.624x。原始记录为
[FlashAttention](../../../runs/bench/infllmv2/w4a16_64k_b1_gpu7_20260819/flash_attn_64k_single_prefill.json)
和 [InfLLM-v2](../../../runs/bench/infllmv2/w4a16_64k_b1_gpu7_20260819/infllmv2_64k_single_prefill.json)，
统一计算结果见 [summary.json](../../../runs/bench/infllmv2/w4a16_sm86_tp1_20260825/summary.json)。

## 验收表

| 任务 | 代码/证据 | 文档编写时状态 |
| --- | --- | --- |
| INF-1：按请求和 head 的 Stage 1 | `infllmv2_update_compressed_k`、`infllmv2_select_blocks`；backend/operator tests | 当前 package 聚焦回归 24/24：backend 15/15、packaged GPU 9/9 |
| INF-2：post-RoPE + paged cache Stage 2 | `decoder_write_cache_with_rope` 后调用 `infllmv2_attention_forward`；顺序与 paged mapping 测试 | 通过：writer 顺序、已写 cache、跨 page 和 paged mapping 均已回归 |
| P0：GQA-tiled Stage 2 | 生产形状 CUDA 路径与 NumPy oracle 算子测试 | 通过；对齐 paged K/V 搬运使当前 65K/8 Stage 2 降至 163.840 us，本地实现为 206.848 us |
| P2：Stage 2 持久化缓冲区 | 后端 workspace 复用与四输出 in-place alias 测试 | 通过；Stage 2 热路径不再调用 `empty`/`full`/`zeros` |
| Stage 1-1：GQA-tiled coarse LSE | split partial/merge kernel、持久化 workspace、语义参考 GPU 测试与 Nsight trace | 通过；32K/1 提升 14.558x，128K/4 提升 21.151x |
| Stage 1-2：精确 radix TopK | 复合 key CUB selector 与 2,048 候选 tie/padding GPU 测试 | 通过；32K/1 提升 2.478x，128K/4 提升 22.545x |
| Stage 1-3：GQA-tiled block-score | 8-block K tile 在 16 个 query heads 间复用；生产 GPU oracle 测试与 Nsight trace | 通过；相对上一检查点，32K/1 提升 4.382x、128K/4 提升 9.391x |
| P3：校准后的长上下文对照 | 当前 CUDA-event 门禁、同形状子仓库复测、Nsight 与完整重启的 65K/8 服务报告 | 算子门禁以 1.5458x 通过；当前完整服务输出吞吐仍为 dense 的 0.983x，因此不声称通用服务加速 |
| 实现文档 | 本文、根目录 `rfc.md` 与当前复测 summary | 已提供可追溯的当前源码结果，2026-08-17 章节取代此前性能检查点 |

## 参考

- [InfLLM-V2 论文](https://arxiv.org/abs/2509.24663)
- [OpenBMB InfLLM-V2 CUDA 实现](https://github.com/OpenBMB/infllmv2_cuda_impl)
- [MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)
- [AngelSlim/Hy3-GGUF](https://huggingface.co/AngelSlim/Hy3-GGUF)
- [llama.cpp PR #25395](https://github.com/ggml-org/llama.cpp/pull/25395)

Hy3-GGUF 和 llama.cpp PR #25395 是用户指定的低比特/Hy3 工程参考，不是本 InfLLM-V2 选块算法或性能数据的来源。
