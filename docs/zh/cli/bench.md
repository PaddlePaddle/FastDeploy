# bench：基准测试
## 1 bench latency：离线延迟测试
### 参数
|参数|说明|默认值|
|-|-|-|
|--input-len|输入序列长度（token）|32|
|--output-len|输出序列长度（token）|128|
|--batch-size|批量大小|8|
|--n|每个提示生成序列数|1|
|--use-beam-search|是否使用束搜索|False|
|--num-iters-warmup|预热迭代次数|10|
|--num-iters|实际测试迭代次数|30|
|--profile|是否进行性能分析|False|
|--output-json|保存延迟结果 JSON 文件路径|None|
|--disable-detokenize|是否禁用 detokenization|False|

### 示例
```
#对推理引擎进行延迟测试
fastdeploy bench latency --model baidu/ERNIE-4.5-0.3B-Paddle
```

## 2 bench serve：在线延迟与吞吐量测试
### 参数
|参数|说明|默认值|
|-|-|-|
|--backend|后端类型|"openai-chat"|
|--base-url|服务器或 API 基础 URL|None|
|--host|主机地址|"127.0.0.1"|
|--port|端口|8000|
|--endpoint|API 路径|"/v1/chat/completions"|
|--model|模型名称|必需|
|--dataset-name|数据集名称|"sharegpt"|
|--dataset-path|数据集路径|None|
|--num-prompts|处理提示数|1000|
|--request-rate|每秒请求数|inf|
|--max-concurrency|最大并发数|None|
|--top-p|采样 top-p (OpenAI 后端)|None|
|--top-k|采样 top-k (OpenAI 后端)|None|
|--temperature|采样温度 (OpenAI 后端)|None|

### 示例
```
#对在线服务进行性能测试
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

## 3 bench throughput：吞吐量测试
### 参数
|参数|说明|默认值|
|-|-|-|
|--backend|推理后端|"fastdeploy"|
|--dataset-name|数据集|"random"|
|--model|模型名称|必需|
|--input-len|输入序列长度|None|
|--output-len|输出序列长度|None|
|--prefix-len|前缀长度|0|
|--n|每个 prompt 生成序列数|1|
|--num-prompts|prompt 数量|50|
|--output-json|保存 JSON 文件路径|None|
|--disable-detokenize|是否禁用 detokenization|False|
|--lora-path|LoRA adapter 路径|None|

### 示例
```
#对推理引擎进行吞吐量测试
fastdeploy bench throughput --model baidu/ERNIE-4.5-0.3B-Paddle \
--backend fastdeploy-chat \
--dataset-name EBChat \
--dataset-path /datasets/filtered_sharedgpt_2000_input_1136_output_200.json \
--max-model-len 32768
```

## 4 bench eval：在线任务效果评估
### 参数
|参数|说明|默认值|
|-|-|-|
|--model, -m|模型名称|"hf"|
|--tasks, -t|任务列表|None|
|--model_args, -a|模型参数|""|
|--num_fewshot, -f|Few-shot 样本数量|None|
|--samples, -E|样本数量|None|
|--batch_size, -b|批量大小|1|
|--device|设备|None|
|--output_path, -o|输出路径|None|
|--write_out, -w|是否写出结果|False|

### 示例
```
#对服务进行相关任务的效果评估
fastdeploy bench eval
--model local-completions
--model_args pretrained=./baidu/ERNIE-4.5-0.3B-Paddle,base_url=http://0.0.0.0:8490/v1/completions
--write_out --tasks ceval-valid_accountant
```
