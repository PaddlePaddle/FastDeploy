# FastDeploy 2.0: 大模型推理部署

FastDeploy升级2.0版本支持多种大模型推理（当前仅支持Qwen2，更多模型即将更新支持)，其推理部署功能涵盖：

- 一行命令即可快速实现模型的服务化部署，并支持流式生成
- 利用张量并行技术加速模型推理
- 支持 PagedAttention 与 continuous batching（动态批处理）
- 兼容 OpenAI 的 HTTP 协议
- 提供 Weight only int8/int4 无损压缩方案
- 支持 Prometheus Metrics 指标

## 环境依赖
- A800/H800/H100
- Python>=3.10
- CUDA>=12.3
- CUDNN>=9.5
- Linux X64

## 安装

推荐使用Docker环境
```
docker pull 
iregistry.baidu-int.com/paddlecloud/base-images:paddlecloud-ubuntu24.04-gcc12.3-cuda12.8-cudnn9.7-openmpi4.1.5-bccl2.15.5.4-ofed24.10-hadoop2.2.4.2-afsshell1.9.3.4095-250227
```

### 源码安装
1. 安装PaddlePaddle GPU(nightly build，代码版本需新于2025.05.30)，详见[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)，指定安装CUDA 12.6 develop(Nightly build)版本，如下命令可完成安装
```
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

2. 安装FastDeploy

```
# git clone FastDeploy仓库
cd FastDeploy
# 一键编译+安装本机可用的sm架构，whl包产物在dist/
bash build.sh
```

## 快速使用

在安装后，执行如下命令快速部署Qwen2模型, 更多参数的配置与含义参考[参数说明](docs/serving.md).

```
# 下载与解压Qwen模型
wget https://fastdeploy.bj.bcebos.com/llm/models/Qwen2-7B-Instruct.tar.gz && tar xvf Qwen2-7B-Instruct.tar.gz
# 指定单卡部署
python -m fastdeploy.entrypoints.openai.api_server --model ./Qwen2-7B-Instruct --port 8188 --tensor-parallel-size 1
```

使用如下命令请求模型服务
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，你的名字是什么？"}
  ]
}'
```
响应结果如下所示
```
{
    "id": "chatcmpl-db662f47-7c8c-4945-9a7a-db563b2ddd8d",
    "object": "chat.completion",
    "created": 1749451045,
    "model": "default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好！我叫通义千问。",
                "reasoning_content": null
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "total_tokens": 35,
        "completion_tokens": 10,
        "prompt_tokens_details": null
    }
}
```
FastDeploy提供与OpenAI完全兼容的服务API(字段`model`与`api_key`目前不支持，设定会被忽略)，用户也可基于openai python api请求服务。

## 部署文档
- [本地部署](docs/offline_inference.md)
- [服务部署](docs/serving.md)
- [服务metrics](docs/metrics.md)
- [调度Scheduler](docs/scheduler.md)

# 代码说明
- [代码目录说明](docs/code_guide.md)
- FastDeploy的使用中存在任何建议和问题，欢迎通过issue反馈。

# 开源说明
FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。 在本项目的开发中，为了对齐[vLLM](https://github.com/vllm-project/vllm)使用接口，参考和直接使用了部分vLLM代码，在此表示感谢。
