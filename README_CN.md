[English](README.md) | 简体中文
<p align="center">
  <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://github.com/user-attachments/assets/42b0039f-39e3-4279-afda-6d1865dfbffb" width="500"></a>
</p>
<p align="center">
    <a href=""><img src="https://img.shields.io/badge/python-3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>

</p>

<p align="center">
     <a href="https://trendshift.io/repositories/4046" target="_blank"><img src="https://trendshift.io/api/badge/repositories/4046" alt="PaddlePaddle%2FFastDeploy | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></br>
    <a href="https://paddlepaddle.github.io/FastDeploy/zh/get_started/installation/nvidia_gpu/"><b> 安装指导 </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/zh/get_started/quick_start"><b> 快速入门 </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/zh/supported_models/"><b> 支持模型列表 </b></a>

</p>

--------------------------------------------------------------------------------
# FastDeploy ：基于飞桨的大语言模型与视觉语言模型推理部署工具包

## 最新活动
**[2025-09] 🔥 FastDeploy v2.2 全新发布: HuggingFace生态模型兼容，性能进一步优化，更新增对[baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)支持!  
**[2025-08] FastDeploy v2.1 发布:全新的KV Cache调度策略，更多模型支持PD分离和CUDA Graph，昆仑、海光等更多硬件支持增强，全方面优化服务和推理引擎的性能。

**[2025-07] 《FastDeploy2.0推理部署实测》专题活动已上线!** 完成文心4.5系列开源模型的推理部署等任务，即可获得骨瓷马克杯等FastDeploy2.0官方周边及丰富奖金！🎁 欢迎大家体验反馈～ 📌[报名地址](https://www.wjx.top/vm/meSsp3L.aspx#)   📌[活动详情](https://github.com/PaddlePaddle/FastDeploy/discussions/2728)

## 关于

**FastDeploy** 是基于飞桨（PaddlePaddle）的大语言模型（LLM）与视觉语言模型（VLM）推理部署工具包，提供**开箱即用的生产级部署方案**，核心技术特性包括：

- 🚀 **负载均衡式PD分解**：工业级解决方案，支持上下文缓存与动态实例角色切换，在保障SLO达标和吞吐量的同时优化资源利用率
- 🔄 **统一KV缓存传输**：轻量级高性能传输库，支持智能NVLink/RDMA选择
- 🤝 **OpenAI API服务与vLLM兼容**：单命令部署，兼容[vLLM](https://github.com/vllm-project/vllm/)接口
- 🧮 **全量化格式支持**：W8A16、W8A8、W4A16、W4A8、W2A16、FP8等
- ⏩ **高级加速技术**：推测解码、多令牌预测（MTP）及分块预填充
- 🖥️ **多硬件支持**：NVIDIA GPU、昆仑芯XPU、海光DCU、昇腾NPU、天数智芯GPU、燧原GCU、沐曦GPU等

## 要求

- 操作系统: Linux
- Python: 3.10 ~ 3.12

## 安装

FastDeploy 支持在**英伟达（NVIDIA）GPU**、**昆仑芯（Kunlunxin）XPU**、**天数（Iluvatar）GPU**、**燧原（Enflame）GCU**、**海光（Hygon）DCU** 以及其他硬件上进行推理部署。详细安装说明如下：

- [英伟达 GPU](./docs/zh/get_started/installation/nvidia_gpu.md)
- [昆仑芯 XPU](./docs/zh/get_started/installation/kunlunxin_xpu.md)
- [天数 CoreX](./docs/zh/get_started/installation/iluvatar_gpu.md)
- [燧原 S60](./docs/zh/get_started/installation/Enflame_gcu.md)
- [海光 DCU](./docs/zh/get_started/installation/hygon_dcu.md)
- [沐曦 GPU](./docs/zh/get_started/installation/metax_gpu.md.md)

**注意:** 我们正在积极拓展硬件支持范围。目前，包括昇腾（Ascend）NPU 等其他硬件平台正在开发测试中。敬请关注更新！

## 入门指南

通过我们的文档了解如何使用 FastDeploy：
- [10分钟快速部署](./docs/zh/get_started/quick_start.md)
- [ERNIE-4.5 部署](./docs/zh/get_started/ernie-4.5.md)
- [ERNIE-4.5-VL 部署](./docs/zh/get_started/ernie-4.5-vl.md)
- [离线推理](./docs/zh/offline_inference.md)
- [在线服务](./docs/zh/online_serving/README.md)
- [最佳实践](./docs/zh/best_practices/README.md)

## 支持模型列表

通过我们的文档了解如何下载模型，如何支持torch格式等：
- [模型支持列表](./docs/zh/supported_models.md)

## 进阶用法

- [量化](./docs/zh/quantization/README.md)
- [分离式部署](./docs/zh/features/disaggregated.md)
- [投机解码](./docs/zh/features/speculative_decoding.md)
- [前缀缓存](./docs/zh/features/prefix_caching.md)
- [分块预填充](./docs/zh/features/chunked_prefill.md)

## 致谢

FastDeploy 依据 [Apache-2.0 开源许可证](./LICENSE). 进行授权。在开发过程中，我们参考并借鉴了 [vLLM](https://github.com/vllm-project/vllm) 的部分代码，以保持接口兼容性，在此表示衷心感谢。
