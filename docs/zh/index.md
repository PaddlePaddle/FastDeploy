# FastDeploy

**FastDeploy** 是基于飞桨（PaddlePaddle）的大语言模型（LLM）与视觉语言模型（VLM）推理部署工具包，提供**开箱即用的生产级部署方案**，核心技术特性包括：

🚀 **负载均衡式PD分解**：工业级解决方案，支持上下文缓存与动态实例角色切换，在保障SLO达标和吞吐量的同时优化资源利用率
🔄 **统一KV缓存传输**：轻量级高性能传输库，支持智能NVLink/RDMA选择
🤝 **OpenAI API服务与vLLM兼容**：单命令部署，兼容[vLLM](https://github.com/vllm-project/vllm/)接口
🧮 **全量化格式支持**：W8A16、W8A8、W4A16、W4A8、W2A16、FP8等
⏩ **高级加速技术**：推测解码、多令牌预测（MTP）及分块预填充
🖥️ **多硬件支持**：NVIDIA GPU、昆仑芯XPU、海光DCU、昇腾NPU、天数智芯GPU、燧原GCU、沐曦GPU等

## 支持模型

| Model | Data Type | PD Disaggregation | Chunked Prefill | Prefix Caching |  MTP | CUDA Graph | Maximum Context Length |
|:--- | :------- | :---------- | :-------- | :-------- | :----- | :----- | :----- |
|ERNIE-4.5-300B-A47B | BF16/WINT4/WINT8/W4A8C8/WINT2/FP8 | ✅| ✅ | ✅|✅(WINT4)| WIP |128K |
|ERNIE-4.5-300B-A47B-Base| BF16/WINT4/WINT8 | ✅| ✅ | ✅|✅(WINT4)| WIP | 128K |
|ERNIE-4.5-VL-424B-A47B | BF16/WINT4/WINT8 | WIP | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-VL-28B-A3B | BF16/WINT4/WINT8 | ❌ | ✅ | WIP | ❌ | WIP |128K |
|ERNIE-4.5-21B-A3B | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-21B-A3B-Base | BF16/WINT4/WINT8/FP8  |  ❌ |  ✅ |  ✅ | WIP | ✅|128K |
|ERNIE-4.5-0.3B | BF16/WINT8/FP8  |  ❌ |  ✅ |  ✅ | ❌ | ✅| 128K |

## 文档说明

本项目文档基于mkdocs支持编译可视化查看，参考如下命令进行编译预览，

```
pip install requirements.txt

cd FastDeploy
mkdocs build

mkdocs serve
```

根据提示打开相应地址即可。
