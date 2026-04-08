[简体中文](../zh/quantization/online_quantization.md)

# Online Quantization

Online quantization refers to the inference engine quantizing weights after loading BF16 weights, rather than loading pre-quantized low-precision weights. FastDeploy supports online quantization of BF16 to various precisions, including: INT4, INT8, and FP8.

## 1. WINT8 & WINT4

Only weights are quantized to INT8 or INT4. During inference, weights are dequantized to BF16 in real-time and then computed with activations.
- **Quantization Granularity**: Only supports channel-wise granularity quantization.
- **Supported Hardware**: GPU, XPU
- **Supported Architecture**: MoE architecture, Dense Linear

### Run WINT8 or WINT4 Inference Service

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

- By specifying `--model baidu/ERNIE-4.5-300B-A47B-Paddle`, the model can be automatically downloaded from AIStudio. FastDeploy depends on Paddle format models. For more information, please refer to [Supported Model List](../supported_models.md).
- By setting `--quantization` to `wint8` or `wint4`, online INT8/INT4 quantization can be selected.
- Deploying ERNIE-4.5-300B-A47B-Paddle WINT8 requires at least 80G *8 cards, while WINT4 requires 80GB* 4 cards.
- For more deployment tutorials, please refer to [get_started](../get_started/ernie-4.5.md).

## 2. Block-wise FP8

Load BF16 model and quantize weights to FP8 numerical type with 128X128 block-wise granularity. During inference, activations are dynamically quantized to FP8 in real-time with token-wise granularity.

- **FP8 Specification**: float8_e4m3fn
- **Supported Hardware**: GPU Hopper architecture
- **Supported Architecture**: MoE architecture, Dense Linear

### Run Block-wise FP8 Inference Service

```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-300B-A47B-Paddle \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --tensor-parallel-size 8 \
       --quantization block_wise_fp8 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

- By specifying `--model baidu/ERNIE-4.5-300B-A47B-Paddle`, the model can be automatically downloaded from AIStudio. FastDeploy depends on Paddle format models. For more information, please refer to [Supported Model List](../supported_models.md).
- By setting `--quantization` to `block_wise_fp8`, online Block-wise FP8 quantization can be selected.
- Deploying ERNIE-4.5-300B-A47B-Paddle Block-wise FP8 requires at least 80G * 8 cards.
- For more deployment tutorials, please refer to [get_started](../get_started/ernie-4.5.md)
