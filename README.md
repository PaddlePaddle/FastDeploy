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
    <a href="docs/get_started/installation/README.md"><b> Installation </b></a>
    |
    <a href="docs/get_started.md"><b> Quick Start </b></a>
    |
    <a href="docs/supported_models.md"><b> Supported Models </b></a>
</p>

--------------------------------------------------------------------------------
# FastDeploy 2.0: Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle

## News

**[2025-06] 🔥 Released FastDeploy v2.0:** Supports inference and deployment for ERNIE 4.5. Furthermore, we open-source an industrial-grade PD disaggregation with context caching, dynamic role switching for effective resource utilization to further enhance inference performance for MoE models.

## About

**FastDeploy** is an inference and deployment toolkit for large language models and visual language models based on PaddlePaddle. It delivers **production-ready, out-of-the-box deployment solutions** with core acceleration technologies:

- 🚀 **Load-Balanced PD Disaggregation**: Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
- 🔄 **Unified KV Cache Transmission**: Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
- 🤝 **OpenAI API Server and vLLM Compatible**: One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
- 🧮 **Comprehensive Quantization Format Support**: W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
- ⏩ **Advanced Acceleration Techniques**: Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
- 🖥️ **Multi-Hardware Support**: NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, Iluvatar GPU, Enflame GCU, MetaX GPU etc.

## Requirements

- OS: Linux
- Python: 3.10 ~ 3.12

## Installation

FastDeploy supports inference deployment on **NVIDIA GPUs**, **Kunlunxin XPUs**, **Iluvatar GPUs**, **Enflame GCUs**, and other hardware. For detailed installation instructions:

- [NVIDIA GPU](./docs/installation/nvidia_cuda.md)
- [Kunlunxin XPU](./docs/en/get_started/installation/kunlunxin_xpu.md)
- [Iluvatar GPU](./docs/en/get_started/installation/iluvatar_gpu.md)
- [Enflame GCU](./docs/en/get_started/installation/Enflame_gcu.md)

**Note:** We are actively working on expanding hardware support. Additional hardware platforms including Ascend NPU, Hygon DCU, and MetaX GPU are currently under development and testing. Stay tuned for updates!

## Get Started

Learn how to use FastDeploy through our documentation:
- [10-Minutes Quick Deployment](./docs/get_started/quick_start.md)
- [ERNIE-4.5 Large Language Model Deployment](./docs/get_started/ernie-4.5.md)
- [ERNIE-4.5-VL Multimodal Model Deployment](./docs/get_started/ernie-4.5-vl.md)
- [Offline Inference Development](./docs/offline_inference.md)
- [Online Service Deployment](./docs/serving/README.md)
- [Full Supported Models List](./docs/supported_models.md)

## Supported Models

| Model | Data Type | PD Disaggregation | Chunked Prefill | Prefix Caching |  MTP | CUDA Graph | Maximum Context Length |
|:--- | :------- | :---------- | :-------- | :-------- | :----- | :----- | :----- |
|ERNIE-4.5-300B-A47B | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8 | ✅（WINT4/W4A8C8/Expert Parallelism)| ✅ | ✅|✅(WINT4)| WIP |128K |
|ERNIE-4.5-300B-A47B-Base| BF16/WINT4/WINT8 | ✅（WINT4/Expert Parallelism)| ✅ | ✅|✅(WINT4)| ❌ | 128K |
|ERNIE-4.5-VL-424B-A47B | BF16/WINT4/WINT8 | WIP | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-VL-28B-A3B | BF16/WINT4/WINT8 | ❌ | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-21B-A3B | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-21B-A3B-Base | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-0.3B | BF16/WINT8/FP8  |  ❌ |  ✅ |  ✅ | ❌ | ✅| 128K |

## Advanced Usage

- [Quantization](./docs/quantization/README.md)
- [PD Disaggregation Deployment](./docs/features/pd_disaggregation.md)
- [Speculative Decoding](./docs/features/speculative_decoding.md)
- [Prefix Caching](./docs/features/prefix_caching.md)
- [Chunked Prefill](./docs/features/chunked_prefill.md)

## Acknowledgement

FastDeploy is licensed under the [Apache-2.0 open-source license](./LICENSE). During development, portions of [vLLM](https://github.com/vllm-project/vllm) code were referenced and incorporated to maintain interface compatibility, for which we express our gratitude.
