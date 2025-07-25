# 🔮 投机解码
本项目基于 PaddlePaddle 实现了高效的 **投机解码（Speculative Decoding）** 推理框架，支持多 Token 预测（Multi-token Proposing, MTP），用于加速大语言模型（LLM）的生成，显著降低时延并提升吞吐量。

## ✅ 投机解码方法支持
### ✅ 支持列表

- **Ngram**

- **MTP (Multi-Token Prediction)**
  - ✅ 已支持：TP 切分
  - ✅ 已支持：共享前缀
  - ✅ 已支持：单机 TP 切分 + PD 分离
  - ⏳ 即将支持：EP + DP + PD 分离
  - ⏳ 即将支持：兼容 Chunk Prefill
  - ⏳ 即将支持：多层 MTP layer

---

### ⏳ 规划中

- Draft Model
- Eagle
- Hydra
- Medusa
- ...

## ⚙️ 高效投机解码框架设计
- **Attention机制**：采用 [Cascade Append Attention](https://flashinfer.ai/2024/02/02/cascade-inference.html) 的 Attention 机制，支持变长查询统一处理，一次前向推理即可完成所有验证。此外，我们对 Kernel 实现进行了深度定制，以最大化 Tensor Core 的利用率，并在高并发场景下仍然保持高吞吐。
- **虚拟填充机制**：采用虚拟填充快速定位输出 Token 的批次 ID，避免了高开销的数据拷贝与切片操作。
- **并行采样与验证**：我们开发了多个融合 Cuda Kernel，用于同时执行采样与验证操作。该 Kernel 支持对每个 batch 样本进行并行处理，避免了显式循环的开销。
- **高效 DraftModel/MTP 框架**：开发多个融合 Cuda Kernel，统一完成模型类方法的前后处理，相比传统的循环、切片方法，性能高效且易维护

## 🔧 参数说明
- `method`: 解码策略，可选值为 `"mtp"` 或 `"ngram"`
- `num_speculative_tokens`: 每轮预测的 Token 数，最大支持 5（当前 MTP 仅支持 1）
- `model`: 若选择 MTP，则需指定 MTP 模型路径
- `quantization`: 模型量化方式，推荐使用 `wint8`
- `batch_size`: 当前支持最大值为 256

## 🚀 使用 Multi-Token-Prediction(MTP) 解码
详见论文：[DeepSeek-V3](https://arxiv.org/pdf/2412.19437)
### TP 并行部署
> 使用 4×H100，量化方式选择 WINT4
> 配置文件：`benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml`

```
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --config ${path_to_FastDeploy}benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}'
```

### PD 分离式部署（1P1D）
> 在8×H100上部署1P1D，P、D节点 分别使用 4×H100；量化方式选择 WINT4
> 与常规 PD 分离部署一致，仅需替换配置文件并新增 speculative_config
详情请参考[PD分离式部署](./disaggregated.md)。
- P 节点（Prefill）

> 配置文件： `benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-prefill.yaml`

```
export FD_LOG_DIR="log_prefill"
rm -rf ${FD_LOG_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m fastdeploy.entrypoints.openai.api_server  \
       --model ${path_to_main_model} \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 \
       --workers 2 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --splitwise-role "prefill" \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000 \
       --scheduler-topic mtp \
       --config ${path_to_FastDeploy}/benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-prefill.yaml \
       --scheduler-password "scheduler_mtp" \
       --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": ""${path_to_mtp_model}"}'  &
```

- D 节点（Decode）

> 配置文件： `benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-decode.yaml`

```
export FD_LOG_DIR="log_prefill"
rm -rf ${FD_LOG_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m fastdeploy.entrypoints.openai.api_server  \
       --model ${path_to_main_model} \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 \
       --workers 2 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --splitwise-role "prefill" \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000 \
       --scheduler-topic mtp \
       --config ${path_to_FastDeploy}/benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-prefill.yaml \
       --scheduler-password "scheduler_mtp" \
       --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": ""${path_to_mtp_model}"}'  &
```

## 🧠 使用 Ngram 解码
该算法通过 n-gram 窗口从 prompt 和已生成的 Token 中进行匹配生成草稿 Token，适合输入和输出有很大 overlap 的场景，如代码续写、文档查询等。
> 使用 4×H100；量化方式选择 WINT4
> 配置文件：benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml

```
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --config ${path_to_FastDeploy}benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml \
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 1, "model": "${mtp_model_path}"}'
```
