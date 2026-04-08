# run-batch: Batch Inference

## Description

Run inference tasks in batch mode. Supports reading input requests from local files or remote URLs, and outputs results to a file or uploads them to a remote destination.

## Usage

```
fastdeploy run-batch --model MODEL --input-file INPUT --output-file OUTPUT [parameters]
```

## Parameters

| Parameter         | Description                                                                                                                                                                                                          | Default |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| -i, --input-file  | Path or URL of the input file. Supports local file paths or HTTP/HTTPS URLs. If a local file path is provided, each line should contain one request. If a URL is provided, the file must be accessible via HTTP GET. | None    |
| -o, --output-file | Path or URL of the output file. Supports local file paths or HTTP/HTTPS URLs. If a local file path is provided, each line will contain one response. If a URL is provided, the file must support HTTP PUT uploads.   | None    |
| --output-tmp-dir  | Temporary directory used to store the output file before uploading it to the output URL.                                                                                                                             | None    |
| --model           | Path to the model                                                                                                                                                                                                    | None    |

For more details on additional parameters, see the [FastDeploy Parameter Documentation](../parameters.md)

## Example

```
fastdeploy run-batch -i Input.json -o Output.json --model baidu/ERNIE-4.5-0.3B-Paddle
```

## Example Input File (Input.json)

```
{"custom_id": "req-00001", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "Tell me a fun fact. (id=1)"}], "temperature": 0.7, "max_tokens": 50}}
{"custom_id": "req-00002", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [{"role": "user", "content": "What's the weather like today? (id=2)"}], "temperature": 0.7, "max_tokens": 50}}
```

## Example Output File (Output.json)

```
{"id":"fastdeploy-84601f40de3e48aeb3fe4d2ca328c32e","custom_id":"req-00001","response":{"status_code":200,"request_id":"fastdeploy-batch-0c18b71f5349453eaf00ae04659a21a0","body":{"id":"chatcmpl-024e9267-3d44-4594-91da-b5033c856da9","object":"chat.completion","created":1761203881,"model":"/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"Here's a fun fact about a funny animal:\n\n**The Elephant in the Room**\n\nElephants are known for their ability to inflate themselves with air pressure. Imagine a giant elephant standing upright, its trunk filling","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":18,"total_tokens":68,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}
{"id":"fastdeploy-04cdfbd5b51e43408971b16be4439888","custom_id":"req-00002","response":{"status_code":200,"request_id":"fastdeploy-batch-dd7a9bebd2964acba6713c5dcb4b4aa6","body":{"id":"chatcmpl-452e9d0b-6c04-4b6f-9a2c-7d961f2dc605","object":"chat.completion","created":1761203881,"model":"/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle","choices":[{"index":0,"message":{"role":"assistant","content":"Based on your query 'What's the weather like today? (id=2)', I need to know what kind of information you are looking for. Do you want to know about a specific weather condition (e.g., sunny, rainy, cloudy) or the current detailed forecast?","multimodal_content":null,"reasoning_content":null,"tool_calls":null,"prompt_token_ids":null,"completion_token_ids":null,"prompt_tokens":null,"completion_tokens":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"total_tokens":70,"completion_tokens":50,"prompt_tokens_details":{"cached_tokens":0}}}},"error":null}
```
