# 采样策略

采样策略用于决定如何从模型的输出概率分布中选择下一个token。FastDeploy目前支持 Top-p 、 Top-k_Top-p 和 Min-p Samping 多种采样策略。

1. Top-p 采样

   * Top-p 采样根据概率累积分布进行截断，仅考虑累计概率达到指定阈值 p 的最可能 token 集合。
   * 动态选择考虑的 token 数量，保证了结果的多样性，同时避免了不太可能的 token。

2. Top-k_top-p 采样

   * 首先进行 top-k 采样，然后在 top-k 的结果上进行归一化，再进行 top-p 采样。
   * 通过限制初始选择范围（top-k）并在其中进行概率累积选择（top-p），提高了生成文本的质量和连贯性。

3. Min-p 采样

   * Min-p 采样首先计算 pivot=max_prob * min_p，然后只保留概率大于pivot的token(其余设置为0)进行后续的采样。
   * 用于过滤掉相对概率过低的token，只从高概率token中采样，提高生成质量。

## 使用说明

在部署时，可以通过设置环境变量 `FD_SAMPLING_CLASS` 来选择采样算法。可选择的值有`base`, `base_non_truncated`, `air`或 `rejection`。

**仅支持 Top-p Sampling 的算法**

* `base`(default)：直接使用 `top_p` 的值进行归一化，倾向于采样概率更大的token。
* `base_non_truncated`：严格按照 Top-p 采样的逻辑执行，首先选择使累积概率达到 `top_p` 的最小集合，然后对这些选择的元素进行归一化。
* `air`：该算法参考 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)的实现，支持 Top-p 采样。

**支持 Top-p 和 Top-k_top-p 采样的算法**

* `rejection`：该算法参考 [flashinfer](https://github.com/flashinfer-ai/flashinfer) 的实现，支持灵活设置 `top_k` 和 `top_p` 参数进行 Top-p 或 Top-k_top-p 采样。

## 配置方式

### Top-p 采样

1. 在部署时，设置环境变量以选择采样算法，默认为base：

```bash
export FD_SAMPLING_CLASS=rejection  # base, base_non_truncated, or air
```

2. 在发送请求时，指定以下参数：

```json
    {
      "top_p": 0.8,
    }
```

### Top-k_top-p 采样

1. 在部署时，设置环境变量以选择rejection采样算法：

```bash
export FD_SAMPLING_CLASS=rejection
```

2. 在发送请求时，指定以下参数：

```json
    {
      "top_p": 0.8,
      "top_k": 20,
    }
```

### Min-p 采样

如果你希望在 top_p 或 top_k_top_p 采样之前使用 min_p 采样，在发送请求时指定以下参数：

```json
    {
      "min_p": 0.1,
    }
```

通过上述配置，你可以根据具体的生成任务需求，灵活选择和使用合适的采样策略。

## 参数说明

* `top_p`: 概率累积分布截断阈值，仅考虑累计概率达到此阈值的最可能token集合。float类型，取值范围为[0.0,1.0]。当top_p=1.0时，考虑所有token；当top_p=0.0时，退化为greedy search。
* `top_k`: 采样概率最高的token数量，考虑概率最高的k个token进行采样范围限制。int类型，取值范围为[0,vocab_size]
* `min_p`：低概率过滤阈值，仅考虑概率大于等于(max_prob*min_p)的token集合。float类型，取值范围为[0.0,1.0]
