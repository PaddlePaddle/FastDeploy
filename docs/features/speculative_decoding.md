[简体中文](../zh/features/speculative_decoding.md)

# 🔮 Speculative Decoding

This project implements an efficient **Speculative Decoding** inference framework based on PaddlePaddle. It supports **Multi-Token Proposing (MTP)** to accelerate large language model (LLM) generation, significantly reducing latency and improving throughput.

---

## ✅ Supported Speculative Decoding Methods

### Supported

- **Naive**: Normal decoding mode that uses the speculative decoding code path without generating draft tokens, useful for testing the speculative decoding framework

- **Ngram**: N-gram matching based speculative decoding

- **Suffix Decoding**

- **MTP (Multi-Token Prediction)**
  - ✅ Supported: TP Sharding
  - ✅ Supported: Shared Prefix
  - ✅ Supported: TP Sharding + PD Separation
  - ⏳ Coming Soon: EP + DP + PD Separation
  - ⏳ Coming Soon: Support Chunk-prefill
  - ⏳ Coming Soon: Multi-layer MTP Layer

- **Decoding with Hybrid MTP and Ngram Methods(Hybrid-MTP-with-Ngram)**

  - Overview: A hybrid method combining MTP and Ngram. First, MTP generates N draft tokens, then Ngram matching is used to supplement additional draft tokens.

  - Use Cases: Suitable when higher draft token coverage is required, leveraging both MTP’s generation capability and the efficiency of Ngram matching.

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

### Basic Parameters

- `method`: The speculative decoding strategy, supports `["mtp", "ngram", "naive", "suffix"]`.
  - `naive`: Normal decoding mode using speculative decoding code path without generating draft tokens
  - `ngram`: N-gram matching based speculative decoding
  - `mtp`: Multi-Token Prediction
  - `suffix`: Suffix decoding based speculative decoding
- `num_speculative_tokens`: Number of speculative tokens to generate; max is 5, currently MTP supports only 1.
- `num_model_steps`: MTP model steps, must satisfy `num_speculative_tokens >= num_model_steps`
- `model`: Path to the MTP draft model when using the `"mtp"` method.
- `quantization`: Quantization method of the MTP model (e.g., WINT4).
- Max `batch_size`: 256

### Verification Strategy (verify_strategy)

Controls how draft tokens are verified:
- `topp` (default): Top-P sampling verification, draft token must be in top-p candidate set
- `greedy`: Greedy verification, draft token must equal target model's argmax output
- `target_match`: Target match verification, draft token must equal target model's sampled output

```bash
--speculative-config '{"method": "mtp", "verify_strategy": "greedy", "num_speculative_tokens": 1, "model": "${path_to_mtp_model}"}'
```

### Accept Policy (accept_policy)

Controls draft token acceptance behavior:
- `normal` (default): Normal verification flow
- `accept_all`: Accept all draft tokens (for debugging)
- `reject_all`: Reject all draft tokens (for debugging)

```bash
--speculative-config '{"method": "mtp", "accept_policy": "accept_all", "num_speculative_tokens": 1}'
```

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
## Decoding with Hybrid MTP and Ngram Methods

When starting the service, you only need to modify the --speculative-config option.
For example, use MTP to generate two draft tokens, and then append three additional draft tokens from Ngram matching:
```
--speculative-config '{"method": "mtp", "num_model_steps": 2, "mtp_strategy": "with_ngram", "num_speculative_tokens": 5, "model": "'$model_path'/mtp"}'
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
    --speculative-config '{"method": "ngram", "num_speculative_tokens": 1}'

```

## 🌲 Using Suffix Decoding

Suffix Decoding is a model-free speculative decoding method that accelerates repetitive inference tasks (e.g., agent workflows, coding) using efficient CPU-based suffix trees for rapid draft token prediction, eliminating GPU overhead.

Run on 4 × H100 GPUs with WINT4 quantization:

> Config file: benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml

```
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --config ${path_to_FastDeploy}benchmarks/yaml/eb45t-32k-wint4-mtp-h100-tp4.yaml \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 4, "suffix_decoding_max_tree_depth": 64, "suffix_decoding_max_cached_requests": 10000, "suffix_decoding_max_spec_factor": 1.0, "suffix_decoding_min_token_prob": 0.1}'
```

Parameter Descriptions

```
# The maximum length of token sequences cached in suffix trees.
self.suffix_decoding_max_tree_depth: int = 64

# The limits of requests that can be stored in the cache.
self.suffix_decoding_max_cached_requests: int = -1

# The factor of matched length, calculated as num_draft_tokens = suffix_max_spec_factor * matched_length
self.suffix_decoding_max_spec_factor: float = 1.0

# The probability threshold for speculated tokens.
self.suffix_decoding_min_token_prob: float = 0.1
```
---

## 📝 Using Naive Mode (Normal Decoding)

Naive mode uses the speculative decoding code path without generating draft tokens, useful for testing the correctness of the speculative decoding framework or establishing performance baselines.

```bash
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${path_to_main_model} \
    --tensor-parallel-size 4 \
    --speculative-config '{"method": "naive", "num_speculative_tokens": 1}'
```

**Note**: In Naive mode, `num_speculative_tokens` will be forced to 0.
