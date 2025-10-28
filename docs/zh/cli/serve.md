# serve：API 服务化
`fastdeploy serve` 提供与 OpenAI 协议兼容的服务化部署。

## 参数
以下是根据您的说明生成的表格：

|选项|说明|默认|
|-|-|-|
|--config|从配置文件读取 CLI 选项（YAML 格式）|None|

更多参数说明见：[FastDeploy 参数文档](../parameters.md)

## 示例
```
# 启动 FastDeploy API 服务器
fastdeploy serve --model baidu/ERNIE-4.5-0.3B-Paddle

# 指定端口启动
fastdeploy serve --model baidu/ERNIE-4.5-0.3B-Paddle --port 8000
```
