# 支持模型列表

FastDeploy目前支持模型列表如下，以下模型提供如下3种下载方式，

- 1. 在FastDeploy部署时，指定 ``model``参数为如下表格中的模型名，即可自动从AIStudio下载模型权重（支持断点续传）
- 2. [HuggingFace/baidu/models](https://huggingface.co/baidu/models) 下载Paddle后缀ERNIE模型，如baidu/ERNIE-4.5-0.3B-Paddle
- 3. [ModelScope/PaddlePaddle](https://www.modelscope.cn/models?name=PaddlePaddle&page=1&tabKey=task) 搜索相应Paddle后缀ERNIE模型，如ERNIE-4.5-0.3B-Paddle

其中第一种方式自动下载时，默认下载路径为 ``~/``(即用户主目录)，用户可以通过配置环境变量 ``FD_MODEL_CACHE``修改默认下载的路径，例如

```
export FD_MODEL_CACHE=/ssd1/download_models
```

| 模型名 | 上下文长度 | 量化方式 | 最小部署资源 | 说明 |
| :----- | :--------------  | :----------- |:----------- |:----------- |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT2 | 1卡*96G显存/1T内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT4 | 4卡*80G显存/1T内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-VL-424B-A47B-Paddle | 32K/128K | WINT8 | 8卡*80G显存/1T内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-300B-A47B-Paddle | 32K/128K | WINT4 | 4卡*64G显存/600G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-300B-A47B-Paddle | 32K/128K | WINT8 | 8卡*64G显存/600G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle | 32K/128K | W4A8C8  | 4卡*64G显存/160G内存 | 限定4卡，建议开启Chunked Prefill |
| baidu/ERNIE-4.5-300B-A47B-FP8-Paddle| 32K/128K | FP8 | 8卡*64G显存/600G内存 | 建议开启Chunked Prefill，仅在PD分离EP并行下支持 |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle | 32K/128K | WINT4 | 4卡*64G显存/600G内存 | 建议开启Chunked Prefill |
| baidu/ERNIE-4.5-300B-A47B-Base-Paddle | 32K/128K | WINT8 | 8卡*64G显存/600G内存 | 建议开启Chunked Prefill |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 32K | WINT4 | 1卡*24G/128G内存 | 需要开启Chunked Prefill |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 128K | WINT4 | 1卡*48G/128G内存 | 需要开启Chunked Prefill |
| baidu/ERNIE-4.5-VL-28B-A3B-Paddle | 32K/128K | WINT8 | 1卡*48G/128G内存 | 需要开启Chunked Prefill |
| baidu/ERNIE-4.5-21B-A3B-Paddle | 32K/128K | WINT4 | 1卡*24G/128G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-21B-A3B-Paddle | 32K/128K | WINT8 | 1卡*48G/128G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle | 32K/128K | WINT4 | 1卡*24G/128G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-21B-A3B-Base-Paddle | 32K/128K | WINT8 | 1卡*48G/128G内存 | 128K需要开启Chunked Prefill |
| baidu/ERNIE-4.5-0.3B-Paddle | 32K/128K | BF16 | 1卡*16G显存/2G内存 | |
| baidu/ERNIE-4.5-0.3B-Base-Paddle | 32K/128K | BF16 | 1卡*16G显存/2G内存 | |

更多模型同步支持中，你可以通过[Github Issues](https://github.com/PaddlePaddle/FastDeploy/issues)向我们提交新模型的支持需求。
