English | [简体中文](README_CN.md)
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
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/"><b> Installation </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/get_started/quick_start"><b> Quick Start </b></a>
    |
    <a href="https://paddlepaddle.github.io/FastDeploy/supported_models/"><b> Supported Models </b></a>

</p>

--------------------------------------------------------------------------------
# FastDeploy : Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle

## News

**[2025-11] FastDeploy v2.3 is newly released!** It adds deployment support for two major models, [ERNIE-4.5-VL-28B-A3B-Thinking](docs/get_started/ernie-4.5-vl-thinking.md) and [PaddleOCR-VL-0.9B](docs/best_practices/PaddleOCR-VL-0.9B.md), across multiple hardware platforms. It further optimizes comprehensive inference performance and brings more deployment features and usability enhancements. For all the upgrade details, refer to the [v2.3 Release Note](https://github.com/PaddlePaddle/FastDeploy/releases/tag/v2.3.0).

**[2025-09] FastDeploy v2.2**: It now offers compatibility with models in the HuggingFace ecosystem, has further optimized performance, and newly adds support for [baidu/ERNIE-21B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-Thinking)!

## About

**FastDeploy** is an inference and deployment toolkit for large language models and visual language models based on PaddlePaddle. It delivers **production-ready, out-of-the-box deployment solutions** with core acceleration technologies:

- 🚀 **Load-Balanced PD Disaggregation**: Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
- 🔄 **Unified KV Cache Transmission**: Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
- 🤝 **OpenAI API Server and vLLM Compatible**: One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
- 🧮 **Comprehensive Quantization Format Support**: W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
- ⏩ **Advanced Acceleration Techniques**: Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
- 🖥️ **Multi-Hardware Support**: NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Iluvatar GPU, Enflame GCU, MetaX GPU, Intel Gaudi etc.

## Requirements

- OS: Linux
- Python: 3.10 ~ 3.12

## Installation

FastDeploy supports inference deployment on **NVIDIA GPUs**, **Kunlunxin XPUs**, **Iluvatar GPUs**, **Enflame GCUs**, **Hygon DCUs** and other hardware. For detailed installation instructions:

- [NVIDIA GPU](./docs/get_started/installation/nvidia_gpu.md)
- [Kunlunxin XPU](./docs/get_started/installation/kunlunxin_xpu.md)
- [Iluvatar GPU](./docs/get_started/installation/iluvatar_gpu.md)
- [Enflame GCU](./docs/get_started/installation/Enflame_gcu.md)
- [Hygon DCU](./docs/get_started/installation/hygon_dcu.md)
- [MetaX GPU](./docs/get_started/installation/metax_gpu.md)
- [Intel Gaudi](./docs/get_started/installation/intel_gaudi.md)

## Get Started

Learn how to use FastDeploy through our documentation:
- [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
- [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
- [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
- [Offline Inference Development](./docs/offline_inference.md)
- [Online Service Deployment](./docs/online_serving/README.md)
- [Best Practices](./docs/best_practices/README.md)

## Supported Models

Learn how to download models, enable using the torch format, and more:
- [Full Supported Models List](./docs/supported_models.md)

## Advanced Usage

- [Quantization](./docs/quantization/README.md)
- [PD Disaggregation Deployment](./docs/features/disaggregated.md)
- [Speculative Decoding](./docs/features/speculative_decoding.md)
- [Prefix Caching](./docs/features/prefix_caching.md)
- [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). During development, portions of [vLLM](https://github.com/vllm-project/vllm) code were referenced and incorporated to maintain interface compatibility, for which we express our gratitude.
