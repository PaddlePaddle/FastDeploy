# chat：对话式生成
`fastdeploy chat` 与正在运行的 API 服务器交互，生成对话。

## 参数
|参数|说明|默认值|
|-|-|-|
|--url|正在运行的 OpenAI-兼容 RESTful API 服务器的 URL|http://localhost:8000/v1|
|--model-name|提示完成中使用的模型名称，默认为列表模型 API 中的第一个模型|None|
|--api-key|用于 OpenAI 服务的 API 密钥，提供时会覆盖环境变量|None|
|--system-prompt|用于在 chat template 中指定 system prompt|None|
|-q, --quick|以 MESSAGE 形式发送单个提示并打印响应，然后退出|None|

## 示例
```
# 直接连接本地主机 API
fastdeploy chat

# 指定 API URL
fastdeploy chat --url http://{fastdeploy-serve-host}:{fastdeploy-serve-port}/v1

# 只需一个提示即可快速聊天
fastdeploy chat --quick "hi"
```
