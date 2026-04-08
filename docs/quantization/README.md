[简体中文](../zh/quantization/README.md)

# Quantization

FastDeploy supports various quantization inference precisions including FP8, INT8, INT4, 2-bits, etc. It supports different precision inference for weights, activations, and KVCache tensors, which can meet the inference requirements of different scenarios such as low cost, low latency, and long context.

## 1. Precision Support List

| Quantization Method | Weight Precision | Activation Precision | KVCache Precision | Online/Offline | Supported Hardware |
|---------|---------|---------|------------|---------|---------|
| [WINT8](online_quantization.md#1-wint8--wint4) | INT8 | BF16 | BF16 | Online |  GPU, XPU |
| [WINT4](online_quantization.md#1-wint8--wint4) | INT4 | BF16 | BF16 | Online | GPU, XPU |
| [Block-wise FP8](online_quantization.md#2-block-wise-fp8) | block-wise static FP8 | token-wise dynamic FP8 | BF16 | Online | GPU |
| [WINT2](wint2.md) | 2Bits | BF16 | BF16 | Offline | GPU |
| MixQuant | INT4/INT8 | INT8/BF16 | INT8/BF16 | Offline | GPU, XPU |

**Notes**

1. **Quantization Method**: Corresponds to the "quantization" field in the quantization configuration file.
2. **Online/Offline Quantization**: Mainly used to distinguish when to quantize the weights.
   - **Online Quantization**: The weights are quantized after being loaded into inference engine.
   - **Offline Quantization**: Before inference, weights are quantized offline and stored as low-bit numerical types. During inference, the quantized low-bit numerical values are loaded.
3. **Dynamic/Static Quantization**: Mainly used to distinguish the quantization method of activations
   - **Static Quantization**: Quantization coefficients are determined and stored before inference. During inference, pre-calculated quantization coefficients are loaded. Since quantization coefficients remain fixed (static) during inference, it's called static quantization.
   - **Dynamic Quantization**: During inference, quantization coefficients for the current batch are calculated in real-time. Since quantization coefficients change dynamically during inference, it's called dynamic quantization.

## 2. Model Support List

| Model Name | Supported Quantization Precision |
|---------|---------|
| ERNIE-4.5-300B-A47B | WINT8, WINT4, Block-wise FP8, MixQuant|

## 3. Quantization Precision Terminology

FastDeploy names various quantization precisions in the following format:

```
{tensor abbreviation}{numerical type}{tensor abbreviation}{numerical type}{tensor abbreviation}{numerical type}
```

Examples:

- **W8A8C8**: W=weights, A=activations, C=CacheKV; 8 defaults to INT8
- **W8A8C16**: 16 defaults to BF16, others same as above
- **W4A16C16 / WInt4 / weight-only int4**: 4 defaults to INT4
- **WNF4A8C8**: NF4 refers to 4bits norm-float numerical type
- **Wfp8Afp8**: Both weights and activations are FP8 precision
- **W4Afp8**: Weights are INT4, activations are FP8
