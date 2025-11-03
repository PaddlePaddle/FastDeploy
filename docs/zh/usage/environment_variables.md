[English](../../usage/environment_variables.md)

# FastDeploy 环境变量说明

FastDeploy 的环境变量保存在了代码库根目录下 fastdeploy/envs.py 文件中，以下是其对应的中文版说明：

```python
environment_variables: dict[str, Callable[[], Any]] = {
    # 构建 FastDeploy 时使用的 CUDA 架构版本，这是一个字符串列表，例如[80,90]
    "FD_BUILDING_ARCS":
    lambda: os.getenv("FD_BUILDING_ARCS", "[]"),

    # 日志目录
    "FD_LOG_DIR":
    lambda: os.getenv("FD_LOG_DIR", "log"),

    # 是否启用调试模式，可设置为 0 或 1
    "FD_DEBUG":
    lambda: int(os.getenv("FD_DEBUG", "0")),

    # FastDeploy 日志保留天数
    "FD_LOG_BACKUP_COUNT":
    lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),

    # 模型下载缓存目录
    "FD_MODEL_CACHE":
    lambda: os.getenv("FD_MODEL_CACHE", None),

    # 停止序列的最大数量
    "FD_MAX_STOP_SEQS_NUM":
    lambda: os.getenv("FD_MAX_STOP_SEQS_NUM", "5"),

    # 停止序列的最大长度
    "FD_STOP_SEQS_MAX_LEN":
    lambda: os.getenv("FD_STOP_SEQS_MAX_LEN", "8"),

    # 将要使用的GPU设备，这是一个用逗号分隔的字符串，例如 0,1,2
    "CUDA_VISIBLE_DEVICES":
    lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),

    # 是否使用 HuggingFace 分词器
    "FD_USE_HF_TOKENIZER":
    lambda: bool(int(os.getenv("FD_USE_HF_TOKENIZER", 0))),

    # 设置 ZMQ 初始化期间接收数据的高水位标记（HWM）
    "FD_ZMQ_SNDHWM":
    lambda: os.getenv("FD_ZMQ_SNDHWM", 10000),

    # 缓存 KV 量化参数的目录
    "FD_CACHE_PARAMS":
    lambda: os.getenv("FD_CACHE_PARAMS", "none"),

    # 设置注意力机制后端，当前可设置为 "NATIVE_ATTN"、"APPEND_ATTN" 或 "MLA_ATTN"
    "FD_ATTENTION_BACKEND":
    lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),

    # 设置采样类别，当前可设置为 "base"、"base_non_truncated"、"air" 或 "rejection"
    "FD_SAMPLING_CLASS":
    lambda: os.getenv("FD_SAMPLING_CLASS", "base"),

    # 设置MoE后端，当前可设置为 "cutlass"、"marlin" 或 "triton"
    "FD_MOE_BACKEND":
    lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),

    # 设置 Triton 内核 JIT 编译目录
    "FD_TRITON_KERNEL_CACHE_DIR":
    lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),

    # 是否从单机 PD 分离转换为集中式推理
    "FD_PD_CHANGEABLE":
    lambda: os.getenv("FD_PD_CHANGEABLE", "1"),

    # 是否使用DeepGemm后端的FP8 blockwise MoE.
    "FD_USE_DEEP_GEMM":
    lambda: bool(int(os.getenv("FD_USE_DEEP_GEMM", "0"))),

    # 是否启用模型权重缓存功能
    "FD_ENABLE_MODEL_LOAD_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_LOAD_CACHE", "0"))),

    # 是否使用 Machete 后端的 wint4 GEMM.
    "FD_USE_MACHETE": lambda: os.getenv("FD_USE_MACHETE", "1"),

    # Used to truncate the string inserted during thinking when reasoning in a model. (</think> for ernie-45-vl, \n</think>\n\n for ernie-x1)
    "FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR": lambda: os.getenv("FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR", "</think>"),

    # cache_transfer_manager 进程残留时退出等待超时时间
    "FD_CACHE_PROC_EXIT_TIMEOUT": lambda: int(os.getenv("FD_CACHE_PROC_EXIT_TIMEOUT", "600")),

    # cache_transfer_manager 进程残留时连续错误阈值
    "FD_CACHE_PROC_ERROR_COUNT": lambda: int(os.getenv("FD_CACHE_PROC_ERROR_COUNT", "10")),}
```
