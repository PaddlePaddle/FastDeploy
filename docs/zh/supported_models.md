# 支持模型列表

FastDeploy目前支持模型列表如下，在FastDeploy部署时，指定 ``model``参数为如下表格中的模型名，即可自动下载模型权重（均支持断点续传），支持如下3种下载源，

- [AIStudio/PaddlePaddle](https://aistudio.baidu.com/modelsoverview) 搜索相应Paddle后缀ERNIE模型，如ERNIE-4.5-0.3B-Paddle
- [ModelScope/PaddlePaddle](https://www.modelscope.cn/models?name=PaddlePaddle&page=1&tabKey=task) 搜索相应Paddle后缀ERNIE模型，如ERNIE-4.5-0.3B-Paddle
- [HuggingFace/baidu/models](https://huggingface.co/baidu/models) 下载Paddle后缀ERNIE模型，如baidu/ERNIE-4.5-0.3B-Paddle

使用自动下载时，默认从AIStudio下载，用户可以通过配置环境变量 ``FD_MODEL_SOURCE``修改默认下载来源，可取值"AISTUDIO"，"MODELSCOPE"或"HUGGINGFACE"；默认下载路径为 ``~/``(即用户主目录)，用户可以通过配置环境变量 ``FD_MODEL_CACHE``修改默认下载的路径，例如

```
export FD_MODEL_SOURCE=AISTUDIO # "AISTUDIO", "MODELSCOPE" or "HUGGINGFACE"
export FD_MODEL_CACHE=/ssd1/download_models
```

> ⭐ **说明**：带星号的模型可直接使用 **HuggingFace Torch 权重**，支持 **FP8/WINT8/WINT4 动态量化** 和 **BF16 精度** 推理，推理时需启用 **`--load_choices "default_v1"`**。

## 纯文本模型

| 模型    | 模型案例                                                     |   Chunked Prefill | Prefix Caching |
|----------|--------------------------------------------------------------|------------------------|------------------|
| ⭐ERNIE | baidu/ERNIE-4.5-300B-A47B-Paddle, baidu/ERNIE-4.5-300B-A47B-PT,<br> baidu/ERNIE-4.5-21B-A3B-PT,baidu/ERNIE-4.5-21B-A3B-Paddle 等      |  ✅              | ✅             |
| ⭐QWEN3-MOE | Qwen/Qwen3-235B-A22B, Qwen/Qwen3-30B-A3B 等            |  ✅              | ✅             |
| ⭐QWEN3    | Qwen/Qwen3-8B, Qwen/Qwen3-4B,Qwen/Qwen3-32B 等               |  ✅              | ✅             |
| ⭐QWEN2    | Qwen/Qwen2-72B, Qwen/Qwen2-7B 等                            |  ✅              | ✅             |
| ⭐QwQ    | Qwen/QwQ-32B 等                          |  ✅              | ✅             |
| ⭐QWEN2.5    | Qwen/Qwen2.5-72B, Qwen/Qwen2.5-14B-Instruct 等                          |  ✅              | ✅             |
| DEEPSEEK | deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1等            |  ✅              | ✅             |

## 多模态语言模型列表

根据模型不同，支持多种模态(文本、图像等)组合：

| 模型    | 模型案例                                                     |  Chunked Prefill | Prefix Caching |
|----------|--------------------------------------------------------------|------------------------|------------------|
| QWEN-VL  | Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct  等    |  ✅              | ✅             |
| ERNIE-VL  | baidu/ERNIE-4.5-VL-424B-A47B-Paddle,baidu/ERNIE-4.5-VL-424B-A47B-PT,<br> baidu/ERNIE-4.5-VL-28B-A3B-Paddle,baidu/ERNIE-4.5-VL-28B-A3B-PT  等  |  ✅              | ✅             |

## 模型规格与部署说明
| 模型名                                      | 上下文长度 | 量化方式 | 最小部署资源          | 说明                                            |
| :------------------------------------------ | :--------- | :------- | :-------------------- | :---------------------------------------------- |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle         | 32K/128K   | WINT4    | 4卡*80G显存/1T内存    | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle         | 32K/128K   | WINT8    | 8卡*80G显存/1T内存    | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-300B-A47B-Paddle            | 32K/128K   | WINT4    | 4卡*64G显存/600G内存  | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-300B-A47B-Paddle            | 32K/128K   | WINT8    | 8卡*64G显存/600G内存  | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle      | 32K/128K   | WINT2    | 1卡*141G显存/600G内存 | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle | 32K/128K   | W4A8C8   | 4卡*64G显存/160G内存  | 限定4卡，建议开启Chunked Prefill                |
| baidu/ERNIE-4.5-300B-A47B-FP8-Paddle        | 32K/128K   | FP8      | 8卡*64G显存/600G内存  | 建议开启Chunked Prefill，仅在PD分离EP并行下支持 |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle       | 32K/128K   | WINT4    | 4卡*64G显存/600G内存  | 建议开启Chunked Prefill                         |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle       | 32K/128K   | WINT8    | 8卡*64G显存/600G内存  | 建议开启Chunked Prefill                         |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle           | 32K        | WINT4    | 1卡*24G/128G内存      | 需要开启Chunked Prefill                         |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle           | 128K       | WINT4    | 1卡*48G/128G内存      | 需要开启Chunked Prefill                         |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle           | 32K/128K   | WINT8    | 1卡*48G/128G内存      | 需要开启Chunked Prefill                         |
| baidu/ERNIE-4.5-21B-A3B-Paddle              | 32K/128K   | WINT4    | 1卡*24G/128G内存      | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-21B-A3B-Paddle              | 32K/128K   | WINT8    | 1卡*48G/128G内存      | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle         | 32K/128K   | WINT4    | 1卡*24G/128G内存      | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle         | 32K/128K   | WINT8    | 1卡*48G/128G内存      | 128K需要开启Chunked Prefill                     |
| baidu/ERNIE-4.5-0.3B-Paddle                 | 32K/128K   | BF16     | 1卡*6G/12G显存/2G内存 |                                                 |
| baidu/ERNIE-4.5-0.3B-Base-Paddle            | 32K/128K   | BF16     | 1卡*6G/12G显存/2G内存 |                                                 |

更多模型同步支持中，你可以通过[Github Issues](https://github.com/PaddlePaddle/FastDeploy/issues)向我们提交新模型的支持需求。
