# 🔮 Speculative Decoding

This project implements an efficient **Speculative Decoding** inference framework based on PaddlePaddle. It supports **Multi-Token Proposing (MTP)** to accelerate large language model (LLM) generation, significantly reducing latency and improving throughput.

---

## ✅ Supported Speculative Decoding Methods

### Supported

- **Ngram**

- **MTP (Multi-Token Prediction)**
  - ✅ Supported: TP Sharding
  - ✅ Supported: Shared Prefix
  - ✅ Supported: TP Sharding + PD Separation
  - ⏳ Coming Soon: EP + DP + PD Separation
  - ⏳ Coming Soon: Support Chunk-prefill
  - ⏳ Coming Soon: Multi-layer MTP Layer

---

### Coming Soon

- Draft Model
- Eagle
- Hydra
- Medusa
- ...

---

## ⚙️ Efficient Speculative Decoding Architecture

- **Attention Mechanism**: We employ [Cascade Append Attention](https://flashinfer.ai/2024/02/02/cascade-inference.html), which allows unified processing of queries with varying token lengths, enabling efficient verification. All tokens can be verified in a single forward pass. We deeply customized the underlying kernels to fully leverage Tensor Cores and maintain high throughput even under heavy concurrency.

- **Virtual Padding Mechanism**: A virtual padding strategy is used to locate output token batch IDs, eliminating the overhead of data copying and slicing operations.

- **Parallel Sampling and Verification**: We developed multiple fused CUDA kernels for concurrent sampling and verification. These kernels allow parallel processing for each sample in a batch, avoiding explicit loop execution on the host side.

- **Efficient Draft Model/MTP Framework**: Multiple fused CUDA kernels are used to handle pre- and post-processing within the model class, replacing traditional loop-based and slicing-based methods with a more performant and maintainable structure.

---

## 🔧 Configuration Parameters

- `method`: The speculative decoding strategy, currently supports `["mtp", "ngram"]`.
- `num_speculative_tokens`: Number of speculative tokens to generate; max is 5, currently MTP supports only 1.
- `model`: Path to the MTP draft model when using the `"mtp"` method.
- `quantization`: Quantization method of the MTP model (e.g., WINT4).
- Max `batch_size`: 256

---

## 🚀 Using Multi-Token Prediction (MTP)

For detailed theory, refer to:
📄 [DeepSeek-V3 Paper](https://arxiv.org/pdf/2412.19437)

### TP Sharding Mode

Launch service on 4 × H100 GPUs using WINT4 quantization (Dense: WINT8, MoE: WINT4):

> Config file: `benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml`

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --config ${path_to_FastDeploy}benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}'
```

### PD-Separated Deployment (1P1D Mode)
Deploy 1P1D on H100 with both Prefill (P) and Decode (D) nodes using TP4 + WINT4 quantization.
This deployment only requires changing the config and adding speculative_config.
For details, refer to the [PD Separation](./disaggregated.md).
- P Node(Prefill)

> Config file: `benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-prefill.yaml`

```
export FD_LOG_DIR="log_prefill"
rm -rf ${FD_LOG_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m fastdeploy.entrypoints.openai.api_server \
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
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}' &
```

- D Node(Decode)

> Config file: `benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-decode.yaml`

```
export FD_LOG_DIR="log_decode"
rm -rf ${FD_LOG_DIR}
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --port 8190 \
    --metrics-port 8191 \
    --engine-worker-queue-port 8192 \
    --cache-queue-port 8193 \
    --workers 2 \
    --tensor-parallel-size 4 \
    --quantization wint4 \
    --splitwise-role "decode" \
    --scheduler-name "splitwise" \
    --scheduler-host "127.0.0.1" \
    --scheduler-port 6379 \
    --scheduler-ttl 9000 \
    --scheduler-topic mtp \
    --config ${path_to_FastDeploy}/benchmarks/yaml/eb45t-32k-wint4-mtp-tp4-decode.yaml \
    --scheduler-password "scheduler_mtp" \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}' &
```

## 🧠 Using Ngram-Based Decoding
This method uses an n-gram sliding window to match the prompt and generated tokens to predict draft tokens. It is particularly effective in scenarios with high input-output overlap (e.g., code completion, document search).

Run on 4 × H100 GPUs with WINT4 quantization:

> Config file: `benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml`

```
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --config ${path_to_FastDeploy}benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "model": "${mtp_model_path}"}'

```
