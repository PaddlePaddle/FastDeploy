[简体中文](../zh/features//pooling_models.md)

# Pooling Models

FastDeploy also supports pooling models, such as embedding models.

In FastDeploy, pooling models implement the `FdModelForPooling` interface.
These models use a `Pooler` to extract the final hidden states of the input
before returning them.

## Configuration

### Model Runner

Run a model in pooling mode via the option `--runner pooling`.

!!! tip<br>
    There is no need to set this option in the vast majority of cases as Fastdeploy can automatically
    detect the appropriate model runner via `--runner auto`.

### Model Conversion

FastDeploy can adapt models for various pooling tasks via the option `--convert <type>`.

If `--runner pooling` has been set (manually or automatically) but the model does not implement the
`FdModelForPooling` interface,
vLLM will attempt to automatically convert the model according to the architecture names
shown in the table below.

| Architecture                                    | `--convert` | Supported pooling tasks               |
|-------------------------------------------------|-------------|---------------------------------------|
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model`  `*ForProcessRewardModel`   | `embed`     |         `embed`                       |

!!! tip<br>
    You can explicitly set `--convert <type>` to specify how to convert the model.

### Pooler Configuration

#### Predefined models

If the `Pooler` defined by the model accepts `pooler_config`,
you can override some of its attributes via the `--pooler-config` option.

#### Converted models

If the model has been converted via `--convert` (see above),
the pooler assigned to each task has the following attributes by default:

| Task       | Pooling Type | Normalization | Softmax |
|------------|--------------|---------------|---------|
| `embed`    | `LAST`       | ✅︎            | ❌      |

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
its Sentence Transformers configuration file (`modules.json`) takes priority over the model's defaults and It can also be specified during model network construction through @default_pooling_type("LAST").

##### Pooling Type

1.LastPool(PoolingType.LAST)

Purpose:Extracts the hidden state of the last token in each sequence

2.AllPool(PoolingType.ALL)

Purpose:Returns the hidden states of all tokens in each sequence

3.CLSPool(PoolingType.CLS)

Purpose:Returns the hidden state of the first token in each sequence (CLS token)

4.MeanPool(PoolingType.MEAN)

Purpose:Computes the average of all token hidden states in each sequence

## Online Serving

FastDeploy's OpenAI-compatible server provides API endpoints and custom reward interfaces.

[Embeddings API], supports text and multi-modal inputs

[Reward API], scores specific content

### Embedding Model:
```python
model_path=Qwen/Qwen3-Embedding-0.6B

python -m fastdeploy.entrypoints.openai.api_server --model ${model_path} \
    --max-num-seqs 256 --max-model-len 32768 \
    --port 9412 --engine-worker-queue-port 7142 \
    --metrics-port 7211 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --runner pooling
```

Request Methods:
A. EmbeddingCompletionRequest Example (Standard Text Input)

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

B. EmbeddingChatRequest Example (Message Sequence Input)

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

### Pooling Model and reward score
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
    --convert embed
```
Request Method: ChatRewardRequest
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
