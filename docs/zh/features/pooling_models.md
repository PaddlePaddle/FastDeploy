[English](../../features/pooling_models.md)

# Pooling Models

FastDeploy也支持pooling模型，例如嵌入(embedding)模型。

在FastDeploy中,池化模型通过`FdModelForPooling`接口。这些模型使用一个`Pooler`来提取输入的最终隐藏状态并返回。

## Configuration

### Model Runner

通过`--runner pooling`选项以池化模型运行模型。

!!! 提示<br>
    在绝大多数情况下无需手动设置该选项，因此Fastdeploy可以通过--runner auto(默认值)自动检测合适的runner。

### Model Conversion

如果模型未实现FdModelForPooling接口但你希望以池化模式运行，FastDeploy可通过`--convert <type>`自动转换模型。

当设置了`--runner pooling`(手动或自动)但模型不符合接口时，FastDeploy会根据模型架构名称自动转换:

| Architecture                                    | `--convert` | 支持的池化类型               |
|-------------------------------------------------|-------------|---------------------------------------|
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model` `**ForProcessRewardModel`  | `embed`     |  `embed`                              |

!!! 提示<br>
    你可以显示设置`--convert <type>`来制定模型转换方式。

### Pooler Configuration

#### Predefined models

如果模型定义的`Pooler`接受pooler_config，你可以通过--pooler_config覆盖部分属性。

#### Converted models

如果模型通过--convert转换，各任务默认的池化配置如下:

| Task       | Pooling Type | Normalization | Softmax |
|------------|--------------|---------------|---------|
| `embed`    | `LAST`       | ✅︎            | ❌      |

加载[Sentence Transformers](https://huggingface.co/sentence-transformers)模型时，其`modules.json`配置优于默认值，也可以通过@default_pooling_type("LAST")在模型组网时指定。

#### Pooling Type

1.LastPool(PoolingType.LAST)

作用:提取每个序列的最后一个token的隐藏状态

2.AllPool(PoolingType.ALL)

作用:返回每个序列的所有token的隐藏状态

3.CLSPool(PoolingType.CLS)

作用:返回每个序列的第一个token(CLS token)的隐藏状态

4.MeanPool(PoolingType.MEAN)

作用:计算每个序列所有token隐藏状态的平均值

## Online Serving

FastDeploy的OpenAI兼容服务器提供了API的端点和自定义的reward接口

- `Embeddings API`，支持文本和多模态输入
- `Reward API`,给指定的内容打分

### Embedding模型:
```python
model_path=Qwen/Qwen3-Embedding-0.6B

python -m fastdeploy.entrypoints.openai.api_server --model ${model_path} \
    --max-num-seqs 256 --max-model-len 32768 \
    --port 9412 --engine-worker-queue-port 7142 \
    --metrics-port 7211 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --runner pooling \

```

请求方式:<br>
A. EmbeddingCompletionRequest 示例（标准文本输入）

```bash
curl -X POST 'YOUR_SERVICE_URL/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-chat-model",
    "input": [
      "This is a sentence for pooling embedding.",
      "Another input text."
    ],
    "user": "test_client"
  }'
```

B. EmbeddingChatRequest 示例（消息序列输入）

```bash
curl -X POST 'YOUR_SERVICE_URL/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "text-embedding-chat-model",
    "messages": [
      {"role": "user", "content": "Generate embedding for user query."}
    ]
  }'
```

### Pooling模型和打分机制
```python
model_path=RM_v1008
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --max-num-seqs 256 \
    --max-model-len 8192 \
    --port 13351 \
    --engine-worker-queue-port 7562 \
    --metrics-port 7531 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --runner pooling \
    --convert embed \
```

请求方式: ChatRewardRequest

```bash
curl --location 'http://xxxx/v1/chat/reward' \
--header 'Content-Type: application/json' \
--data '{
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://xxx/a.png"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "图里有几个人"
        }
      ]
    }
  ],
  "user": "user-123",
  "chat_template": null,
  "chat_template_kwargs": {
    "custom_var": "value"
  },
  "mm_processor_kwargs": {
    "image_size": 224
  }
}'
```
