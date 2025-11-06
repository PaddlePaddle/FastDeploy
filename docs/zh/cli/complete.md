# complete：补全式生成
`fastdeploy complete` 根据给定提示词生成文本完成。

## 参数
|参数|说明|默认值|
|-|-|-|
|--url|正在运行的 OpenAI-兼容 RESTful API 服务器的 URL|http://localhost:8000/v1|
|--model-name|提示完成中使用的模型名称|None|
|--api-key|用于 OpenAI 服务的 API 密钥|None|
|--system-prompt|在 chat template 中指定 system prompt|None|
|-q, --quick|以 MESSAGE 形式发送单个提示并打印响应|None|

## 示例
```
# 直接连接本地主机 API
fastdeploy complete

# 指定 API URL
fastdeploy complete --url http://{fastdeploy-serve-host}:{fastdeploy-serve-port}/v1

# 快速完成
fastdeploy complete --quick "The future of AI is"
```
