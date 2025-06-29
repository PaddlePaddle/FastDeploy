# FastDeploy

**FastDeploy** is an inference and deployment toolkit for large language models and visual language models based on PaddlePaddle. It delivers **production-ready, out-of-the-box deployment solutions** with core acceleration technologies:

- 🚀 **Load-Balanced PD Disaggregation**: Industrial-grade solution featuring context caching and dynamic instance role switching. Optimizes resource utilization while balancing SLO compliance and throughput.
- 🔄 **Unified KV Cache Transmission**: Lightweight high-performance transport library with intelligent NVLink/RDMA selection.
- 🤝 **OpenAI API Server and vLLM Compatible**: One-command deployment with [vLLM](https://github.com/vllm-project/vllm/) interface compatibility.
- 🧮 **Comprehensive Quantization Format Support**: W8A16, W8A8, W4A16, W4A8, W2A16, FP8, and more.
- ⏩ **Advanced Acceleration Techniques**: Speculative decoding, Multi-Token Prediction (MTP) and Chunked Prefill.
- 🖥️ **Multi-Hardware Support**: NVIDIA GPU, Kunlunxin XPU, Hygon DCU, Ascend NPU, etc.

## Supported Models

| Model | Data Type | PD Disaggregation | Chunked Prefill | Prefix Caching |  MTP | CUDA Graph | Maximum Context Length |
|:--- | :------- | :---------- | :-------- | :-------- | :----- | :----- | :----- |
|ERNIE-4.5-300B-A47B | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8 | ✅（WINT4/W4A8C8/Expert Parallelism)| ✅ | ✅|✅(WINT4)| WIP |128K |
|ERNIE-4.5-300B-A47B-Base| BF16/WINT4/WINT8 | ✅（WINT4/Expert Parallelism)| ✅ | ✅|✅(WINT4)| ❌ | 128K |
|ERNIE-4.5-VL-424B-A47B | BF16/WINT4/WINT8 | WIP | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-VL-28B-A3B | BF16/WINT4/WINT8 | ❌ | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-21B-A3B | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-21B-A3B-Base | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-0.3B | BF16/WINT8/FP8  |  ❌ |  ✅ |  ✅ | ❌ | ✅|128K |

## Documentation

This project's documentation supports visual compilation via mkdocs. Use the following commands to compile and preview:

```bash
pip install requirements.txt

cd FastDeploy
mkdocs build

mkdocs serve
```

Open the indicated address to view the documentation.
