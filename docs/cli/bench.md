# bench: Benchmark Testing

## 1. bench latency: Offline Latency Test

### Parameters

| Parameter            | Description                                 | Default |
| -------------------- | ------------------------------------------- | ------- |
| --input-len          | Input sequence length (tokens)              | 32      |
| --output-len         | Output sequence length (tokens)             | 128     |
| --batch-size         | Batch size                                  | 8       |
| --n                  | Number of sequences generated per prompt    | 1       |
| --use-beam-search    | Whether to use beam search                  | False   |
| --num-iters-warmup   | Number of warmup iterations                 | 10      |
| --num-iters          | Number of actual test iterations            | 30      |
| --profile            | Whether to enable performance profiling     | False   |
| --output-json        | Path to save latency results as a JSON file | None    |
| --disable-detokenize | Whether to disable detokenization           | False   |

### Example

```
# Run latency benchmark on the inference engine
fastdeploy bench latency --model baidu/ERNIE-4.5-0.3B-Paddle
```

## 2. bench serve: Online Latency and Throughput Test

### Parameters

| Parameter         | Description                           | Default                |
| ----------------- | ------------------------------------- | ---------------------- |
| --backend         | Backend type                          | "openai-chat"          |
| --base-url        | Base URL of the server or API         | None                   |
| --host            | Host address                          | "127.0.0.1"            |
| --port            | Port                                  | 8000                   |
| --endpoint        | API endpoint path                     | "/v1/chat/completions" |
| --model           | Model name                            | Required               |
| --dataset-name    | Dataset name                          | "sharegpt"             |
| --dataset-path    | Path to dataset                       | None                   |
| --num-prompts     | Number of prompts to process          | 1000                   |
| --request-rate    | Requests per second                   | inf                    |
| --max-concurrency | Maximum concurrency                   | None                   |
| --top-p           | Sampling top-p (OpenAI backend)       | None                   |
| --top-k           | Sampling top-k (OpenAI backend)       | None                   |
| --temperature     | Sampling temperature (OpenAI backend) | None                   |

### Example

```
# Run online performance test
fastdeploy bench serve --backend openai-chat \
  --model baidu/ERNIE-4.5-0.3B-Paddle \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 8891 \
  --dataset-name EBChat \
  --dataset-path /datasets/filtered_sharedgpt_2000_input_1136_output_200.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --save-result
```

## 3. bench throughput: Throughput Test

### Parameters

| Parameter            | Description                              | Default      |
| -------------------- | ---------------------------------------- | ------------ |
| --backend            | Inference backend                        | "fastdeploy" |
| --dataset-name       | Dataset name                             | "random"     |
| --model              | Model name                               | Required     |
| --input-len          | Input sequence length                    | None         |
| --output-len         | Output sequence length                   | None         |
| --prefix-len         | Prefix length                            | 0            |
| --n                  | Number of sequences generated per prompt | 1            |
| --num-prompts        | Number of prompts                        | 50           |
| --output-json        | Path to save results as a JSON file      | None         |
| --disable-detokenize | Whether to disable detokenization        | False        |
| --lora-path          | Path to LoRA adapter                     | None         |

### Example

```
# Run throughput benchmark on the inference engine
fastdeploy bench throughput --model baidu/ERNIE-4.5-0.3B-Paddle \
--backend fastdeploy-chat \
--dataset-name EBChat \
--dataset-path /datasets/filtered_sharedgpt_2000_input_1136_output_200.json \
--max-model-len 32768
```

## 4. bench eval: Online Task Evaluation

### Parameters

| Parameter         | Description                     | Default |
| ----------------- | ------------------------------- | ------- |
| --model, -m       | Model name                      | "hf"    |
| --tasks, -t       | List of evaluation tasks        | None    |
| --model_args, -a  | Model arguments                 | ""      |
| --num_fewshot, -f | Number of few-shot examples     | None    |
| --samples, -E     | Number of samples               | None    |
| --batch_size, -b  | Batch size                      | 1       |
| --device          | Device                          | None    |
| --output_path, -o | Output file path                | None    |
| --write_out, -w   | Whether to write output results | False   |

### Example

```
# Run task evaluation on an online service
fastdeploy bench eval --model local-completions \
  --model_args pretrained=./baidu/ERNIE-4.5-0.3B-Paddle,base_url=http://0.0.0.0:8490/v1/completions
  --write_out \
  --tasks ceval-valid_accountant
```
