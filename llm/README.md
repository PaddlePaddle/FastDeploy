
<h1 align="center"><b><em>飞桨大模型高性能部署工具FastDeploy</em></b></h1>

*FastDeploy基于英伟达Triton框架专为服务器场景的大模型服务化部署而设计的解决方案。它提供了支持gRPC、HTTP协议的服务接口，以及流式Token输出能力。底层推理引擎支持连续批处理、weight only int8、后训练量化（PTQ）等加速优化策略，为用户带来易用且高性能的部署体验。*

# 快速开始

  基于预编译镜像部署，本节以 Meta-Llama-3-8B-Instruct-A8W8C8 为例，更多模型请参考[LLaMA](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)、[Qwen](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/qwen.md)、[Mixtral](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/mixtral.md), 更细致的模型推理、量化教程可以参考[大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)：

  ```
    # 下载模型
    wget https://paddle-qa.bj.bcebos.com/inference_model/Meta-Llama-3-8B-Instruct-A8W8C8.tar
    mkdir Llama-3-8B-A8W8C8 && tar -xf Meta-Llama-3-8B-Instruct-A8W8C8.tar -C Llama-3-8B-A8W8C8

    # 挂载模型文件
    export MODEL_PATH=${PWD}/Llama-3-8B-A8W8C8

    docker run --gpus all --shm-size 5G --network=host \
    -v ${MODEL_PATH}:/models/ \
    -dit registry.baidubce.com/paddlepaddle/fastdeploy:llm-serving-cuda123-cudnn9-v1.0 \
    bash -c 'export USE_CACHE_KV_INT8=1 && cd /opt/output/Serving && bash start_server.sh; exec bash'
  ```

  等待服务启动成功（服务初次启动大概需要40s），可以通过以下命令测试：

  ```
    curl 127.0.0.1:9965/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"text": "hello, llm"}'
  ```

Note:
1. 请保证 shm-size >= 5，不然可能会导致服务启动失败

更多关于 FastDeploy 的使用方法，请查看[服务化部署流程](https://console.cloud.baidu-int.com/devops/icode/repos/baidu/fastdeploy/serving/blob/opensource/docs/FastDeploy_usage_tutorial.md)

# benchmark 测试

我们在 `Llama-3-8B-Instruct` 模型不同的精度下，对 FastDeploy 的性能进行测试，测试结果如下表所示：

<table align="center" border="1" style="text-align: center; vertical-align: middle;">
    <tr>
        <th align="center">框架</th>
        <th align="center">精度</th>
        <th align="center">QPS</th>
        <th align="center">tokens/s</th>
        <th align="center">整句时延</th>
    </tr>
    <tr>
        <td rowspan="3">FastDeploy</td>
        <td>FP16/BF16</td>
        <td>16.21</td>
        <td>3171.09</td>
        <td>7.15</td>
    </tr>
    <tr>
        <td>WINT8</td>
        <td>14.84</td>
        <td>2906.27</td>
        <td>7.95</td>
    </tr>
    <tr>
        <td>W8A8C8-INT8</td>
        <td>20.60</td>
        <td>4031.75</td>
        <td>5.61</td>
    </tr>
    <tr>
        <td rowspan="3">vLLM</td>
        <td>FP16/BF16</td>
        <td>9.07</td>
        <td>1766.11</td>
        <td>13.32</td>
    </tr>
    <tr>
        <td>WINT8</td>
        <td>8.23</td>
        <td>1602.96</td>
        <td>14.85</td>
    </tr>
    <tr>
        <td>W8A8C8-INT8</td>
        <td>9.41</td>
        <td>1831.81</td>
        <td>12.76</td>
    </tr>
</table>

- 测试环境：
  - GPU：NVIDIA A100-SXM4-80GB
  - cuda 版本：11.6
  - cudnn 版本：8.4.0
  - Batch Size: 128
  - 请求并发量：128
  - vLLM 版本：v0.5.3
  - TRT-LLM 版本：v0.11.0
  - 数据集：[ShareGPT_V3_unfiltered_cleaned_split.json](https://huggingface.co/datasets/learnanything/sharegpt_v3_unfiltered_cleaned_split/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json)

# License

FastDeploy 遵循 [Apache-2.0开源协议](https://github.com/PaddlePaddle/FastDeploy/blob/develop/LICENSE) 。
