[English](../supported_models.md)

# 支持模型列表

FastDeploy目前支持模型列表如下，在FastDeploy部署时，指定 ``model``参数为如下表格中的模型名，即可自动下载模型权重（均支持断点续传），支持如下3种下载源，

- [AIStudio](https://aistudio.baidu.com/modelsoverview)
- [ModelScope](https://www.modelscope.cn/models)
- [HuggingFace](https://huggingface.co/models)

使用自动下载时，默认从AIStudio下载，用户可以通过配置环境变量 ``FD_MODEL_SOURCE``修改默认下载来源，可取值"AISTUDIO"，"MODELSCOPE"或"HUGGINGFACE"；默认下载路径为 ``~/``(即用户主目录)，用户可以通过配置环境变量 ``FD_MODEL_CACHE``修改默认下载的路径，例如

```
export FD_MODEL_SOURCE=AISTUDIO # "AISTUDIO", "MODELSCOPE" or "HUGGINGFACE"
export FD_MODEL_CACHE=/ssd1/download_models
```

> 以baidu/ERNIE-4.5-21B-A3B-PT为例启动命令如下
```
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-0.3B-PT \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \
       --max-num-seqs 32
```

## 纯文本模型列表

|模型|DataType|模型案例|
|-|-|-|
|⭐ERNIE|BF16\WINT4\WINT8\W4A8C8\WINT2\FP8|baidu/ERNIE-4.5-VL-424B-A47B-Paddle;<br>baidu/ERNIE-4.5-300B-A47B-Paddle<br>&emsp;[快速部署](./get_started/ernie-4.5.md) &emsp; [最佳实践](./best_practices/ERNIE-4.5-300B-A47B-Paddle.md);<br>baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle;<br>baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle;<br>baidu/ERNIE-4.5-300B-A47B-FP8-Paddle;<br>baidu/ERNIE-4.5-300B-A47B-Base-Paddle;<br>[baidu/ERNIE-4.5-21B-A3B-Paddle](./best_practices/ERNIE-4.5-21B-A3B-Paddle.md);<br>baidu/ERNIE-4.5-21B-A3B-Base-Paddle;<br>baidu/ERNIE-4.5-21B-A3B-Thinking;<br>[baidu/ERNIE-4.5-VL-28B-A3B-Thinking](./get_started/ernie-4.5-vl-thinking.md);<br>baidu/ERNIE-4.5-0.3B-Paddle<br>&emsp;[快速部署](./get_started/quick_start.md) &emsp; [最佳实践](./best_practices/ERNIE-4.5-0.3B-Paddle.md);<br>baidu/ERNIE-4.5-0.3B-Base-Paddle, etc.|
|⭐QWEN3-MOE|BF16/WINT4/WINT8/FP8|Qwen/Qwen3-235B-A22B;<br>Qwen/Qwen3-30B-A3B, etc.|
|⭐QWEN3|BF16/WINT8/FP8|Qwen/qwen3-32B;<br>Qwen/qwen3-14B;<br>Qwen/qwen3-8B;<br>Qwen/qwen3-4B;<br>Qwen/qwen3-1.7B;<br>[Qwen/qwen3-0.6B](./get_started/quick_start_qwen.md), etc.|
|⭐QWEN2.5|BF16/WINT8/FP8|Qwen/qwen2.5-72B;<br>Qwen/qwen2.5-32B;<br>Qwen/qwen2.5-14B;<br>Qwen/qwen2.5-7B;<br>Qwen/qwen2.5-3B;<br>Qwen/qwen2.5-1.5B;<br>Qwen/qwen2.5-0.5B, etc.|
|⭐QWEN2|BF16/WINT8/FP8|Qwen/Qwen/qwen2-72B;<br>Qwen/Qwen/qwen2-7B;<br>Qwen/qwen2-1.5B;<br>Qwen/qwen2-0.5B;<br>Qwen/QwQ-32, etc.|
|⭐DEEPSEEK|BF16/WINT4|unsloth/DeepSeek-V3.1-BF16;<br>unsloth/DeepSeek-V3-0324-BF16;<br>unsloth/DeepSeek-R1-BF16, etc.|
|⭐GPT-OSS|BF16/WINT8|unsloth/gpt-oss-20b-BF16, etc.|
|⭐GLM-4.5/4.6|BF16/wfp8afp8|zai-org/GLM-4.5-Air;<br>zai-org/GLM-4.6<br>&emsp;[最佳实践](./best_practices/GLM-4-MoE-Text.md) etc.|

## 多模态语言模型列表

根据模型不同，支持多种模态(文本、图像等)组合：

|模型|DataType|模型案例|
|-|-|-|
| ERNIE-VL  |BF16/WINT4/WINT8| baidu/ERNIE-4.5-VL-424B-A47B-Paddle<br>&emsp;[快速部署](./get_started/ernie-4.5-vl.md) &emsp; [最佳实践](./best_practices/ERNIE-4.5-VL-424B-A47B-Paddle.md) ;<br>baidu/ERNIE-4.5-VL-28B-A3B-Paddle<br>&emsp;[快速部署](./get_started/quick_start_vl.md) &emsp; [最佳实践](./best_practices/ERNIE-4.5-VL-28B-A3B-Paddle.md) ;<br>baidu/ERNIE-4.5-VL-28B-A3B-Thinking<br>&emsp;[快速部署](./get_started/ernie-4.5-vl-thinking.md)&emsp; [最佳实践](./best_practices/ERNIE-4.5-VL-28B-A3B-Thinking.md) ;
| PaddleOCR-VL  |BF16/WINT4/WINT8| PaddlePaddle/PaddleOCR-VL<br>&emsp; [最佳实践](./best_practices/PaddleOCR-VL-0.9B.md) ;|
| QWEN-VL  |BF16/WINT4/FP8| Qwen/Qwen2.5-VL-72B-Instruct;<br>Qwen/Qwen2.5-VL-32B-Instruct;<br>Qwen/Qwen2.5-VL-7B-Instruct;<br>Qwen/Qwen2.5-VL-3B-Instruct|

更多模型同步支持中，你可以通过[Github Issues](https://github.com/PaddlePaddle/FastDeploy/issues)向我们提交新模型的支持需求。
