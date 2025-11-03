# run-batch：批处理
## 说明
批量运行推理任务。支持从本地文件或远程 URL 读取输入请求，并将推理结果输出到文件或上传至远程 URL

## 用法
```
  fastdeploy run-batch --model MODEL --input-file INPUT --output-file OUTPUT [参数]
```

## 参数
|参数|说明|默认值|
|-|-|-|
|-i, --input-file|单个输入文件的路径或 URL。目前支持本地文件路径或 HTTP/HTTPS 协议。如果是文件路径，文件里每行一个请求；如果指定了 URL，文件应可通过 HTTP GET 访问。|None|
|-o, --output-file|单个输出文件的路径或 URL。目前支持本地文件路径或网络 URL（HTTP/HTTPS）。如果是输出文件路径，每行一个响应；如果指定了 URL，文件应可通过 HTTP PUT 上传。|None|
|--output-tmp-dir|在将输出文件上传到输出 URL 之前，用于存放输出文件的临时目录。|None|
|--model|模型路径|None|

更多参数说明见：[FastDeploy 参数文档](../parameters.md)

## 示例
```
fastdeploy run-batch -i Input.json -o Output.json --model baidu/ERNIE-4.5-0.3B-Paddle
```

## 输入文件格式示例（Input.json）
```
{"custom_id": "req-00001", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Tell me a fun fact. (id=1)"}], "temperature": 0.7, "max_tokens": 50}}
{"custom_id": "req-00002", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "What's the weather like today? (id=2)"}], "temperature": 0.7, "max_tokens": 50}}
```

## 输出文件格式示例（Output.json）
```
{"id":"fastdeploy-84601f40de3e48aeb3fe4d2ca328c32e","custom_id":"req-00001","response":{"status_code":200,"request_id":"fastdeploy-batch-0c18b71f5349453eaf00ae04659a21a0","body":{"id":"chatcmpl-024e9267-3d44-4594-91da-b5033c856da9","object":"chat.completion","created":1761203881,"model":"/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"Here's a fun fact about a funny animal:\n\n**The Elephant in the Room**\n\nElephants are known for their ability to inflate themselves with air pressure. Imagine a giant elephant standing upright, its trunk filling","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"total_tokens":68,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}
{"id":"fastdeploy-04cdfbd5b51e43408971b16be4439888","custom_id":"req-00002","response":{"status_code":200,"request_id":"fastdeploy-batch-dd7a9bebd2964acba6713c5dcb4b4aa6","body":{"id":"chatcmpl-452e9d0b-6c04-4b6f-9a2c-7d961f2dc605","object":"chat.completion","created":1761203881,"model":"/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"根据您提供的查询语句“What's the weather like today? (id=2)”，我需要先了解您想要查询的内容。请问您想查询的是某个特定天气状况（如晴天、下雨、阴天等）还是当前具体","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"total_tokens":70,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}

```
