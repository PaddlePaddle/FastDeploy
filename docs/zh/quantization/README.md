[English](../../quantization/README.md)

# 量化

FastDeploy支持FP8、INT8、INT4、2-bit等多种量化推理精度，支持模型权重、激活和KVCache 3种张量的不同精度推理，可以满足低成本、低时延、长上下文等不同场景的推理需求。

## 1. 精度支持列表

| 量化方法 | 权重精度 | 激活精度 | KVCache精度 | 在线/离线 | 支持硬件 |
|---------|---------|---------|------------|---------|---------|
| [WINT8](online_quantization.md#1-wint8--wint4) | INT8 | BF16 | BF16 | 在线 |  GPU, XPU |
| [WINT4](online_quantization.md#1-wint8--wint4) | INT4 | BF16 | BF16 | 在线 | GPU, XPU |
| [block_wise_fp8](online_quantization.md#2-block-wise-fp8) | block-wise static FP8 | token-wise dynamic FP8 | BF16 | 在线 | GPU |
| [WINT2](wint2.md) | 2Bits | BF16 | BF16 | 离线 | GPU |
| MixQuant | INT4/INT8 | INT8/BF16 | INT8/BF16 | 离线 | GPU, XPU |

**说明**

1. **量化方法**：对应量化配置文件中的"quantization"字段；
2. **在线/离线量化**：主要用于区分权重的量化时间
   - **在线量化**：推理引擎在加载 BF16 权重后，再对权重做量化；
   - **离线量化**：在推理之前，将权重离线地量化并存储为低比特数值类型，推理时，加载量化后的低比特数值。
3. **动态量化/静态量化**：主要用于区别激活的量化方式
   - **静态量化（static）**：在推理之前，确定并存储量化系数，推理时加载提前计算好的量化系数。因为量化系数在推理时是固定不变的，所以叫静态量化（static quantization）；
   - **动态量化（dynamic）**：在推理时，即时地统计当前batch的量化系数。因为量化系数在推理时是动态地变化的，所以叫动态量化（dynamic quantization）。

## 2. 模型支持列表

| 模型名称 | 支持量化精度 |
|---------|---------|
| ERNIE-4.5-300B-A47B | WINT8, WINT4, Block_wise= FP8, MixQuant|

## 3. 量化精度术语

FastDeploy 按以下格式命名各种量化精度：

```
{tensor缩写}{数值类型}{tensor缩写}{数值类型}{tensor缩写}{数值类型}
```

部分示例如下：

- **W8A8C8**：W=weights，A=activations，C=CacheKV；8默认为INT8
- **W8A8C16**：16默认为BF16，其它同上
- **W4A16C16 / WInt4 / weight-only int4**：4默认为INT4
- **WNF4A8C8**：NF4指4bit norm-float数值类型
- **Wfp8Afp8**：权重和激活均为FP8精度
- **W4Afp8**：权重为INT4, 激活为FP8
