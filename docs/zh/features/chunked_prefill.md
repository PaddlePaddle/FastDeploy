[English](../../features/chunked_prefill.md)

# Chunked Prefill 与 128K 长文推理部署

Chunked Prefill 采用分块策略，将预填充（Prefill）阶段请求拆解为小规模子任务，与解码（Decode）请求混合批处理执行。可以更好地平衡计算密集型（Prefill）和访存密集型（Decode）操作，优化GPU资源利用率，减少单次Prefill的计算量和显存占用，从而降低显存峰值，避免显存不足的问题。

在启用 Chunked Prefill 机制后，调度策略采用以下优先级规则：

- 解码请求优先处理：系统会优先将所有待处理的解码请求（Decode）进行批量整合，确保生成类任务获得即时响应能力。
- 弹性分块机制：当存在等待处理的预填充任务时，系统会在 `max_num_batched_tokens` 预算范围内（即当前批次可容纳的最大 token 数量）调度预填充（Prefill）操作。若单个预填充请求的 token 量超过剩余预算容量，系统会自动将其拆分为多个符合预算限制的子块（Chunks）。

通过 Chunked Prefill 机制，可以有效降低 Token 间时延（Inter-Token Latency），同时优化显存占用，支持更大长度的输入。

更多信息请查阅相关论文 [https://arxiv.org/pdf/2308.16369](https://arxiv.org/pdf/2308.16369)。

## 使用与调优

在 FastDeploy 中开启 Chunked Prefill 时，可以通过 `max_num_batched_tokens` 参数调节服务性能：

- 较小的 `max_num_batched_tokens` 可以获得更好的 Token 间时延（Inter-Token Latency）。
- 更大的 `max_num_batched_tokens` 可以获得更好的首 Token 时延（Time To First Token）。
- 建议配置 `max_num_batched_tokens > 8096`，当在算力充足的硬件上部署小模型时，可以进一步增大 `max_num_batched_tokens` 来获得更高的吞吐量。

## 适用场景

1. 长文推理部署，如 128K 长文推理部署等场景。可以通过 Chunked prefill 技术降低显存峰值，避免显存不足的问题。
2. 生成类任务，通过 Chunked prefill 技术可以降低 Token 间时延（Inter-Token Latency），提高生成类任务的响应速度。
