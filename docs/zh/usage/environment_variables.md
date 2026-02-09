[English](../../usage/environment_variables.md)

# FastDeploy 环境变量说明

FastDeploy 的环境变量保存在了代码库根目录下 fastdeploy/envs.py 文件中，以下是其对应的中文版说明：

```python
environment_variables: dict[str, Callable[[], Any]] = {
    # 是否在 CPU 上使用 BF16
    "FD_CPU_USE_BF16": lambda: os.getenv("FD_CPU_USE_BF16", "False"),

    # 构建 FastDeploy 时使用的 CUDA 架构版本，这是一个字符串列表，例如[80,90]
    "FD_BUILDING_ARCS": lambda: os.getenv("FD_BUILDING_ARCS", "[]"),

    # 日志目录
    "FD_LOG_DIR": lambda: os.getenv("FD_LOG_DIR", "log"),

    # 是否启用调试模式，可设置为 0 或 1
    "FD_DEBUG": lambda: int(os.getenv("FD_DEBUG", "0")),

    # FastDeploy 日志保留天数
    "FD_LOG_BACKUP_COUNT": lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),

    # 模型下载源，可设置为 "AISTUDIO"、"MODELSCOPE" 或 "HUGGINGFACE"
    "FD_MODEL_SOURCE": lambda: os.getenv("FD_MODEL_SOURCE", "AISTUDIO"),

    # 模型下载缓存目录
    "FD_MODEL_CACHE": lambda: os.getenv("FD_MODEL_CACHE", None),

    # 停止序列的最大数量
    "FD_MAX_STOP_SEQS_NUM": lambda: int(os.getenv("FD_MAX_STOP_SEQS_NUM", "5")),

    # 停止序列的最大长度
    "FD_STOP_SEQS_MAX_LEN": lambda: int(os.getenv("FD_STOP_SEQS_MAX_LEN", "8")),

    # 将要使用的GPU设备，这是一个用逗号分隔的字符串，例如 0,1,2
    "CUDA_VISIBLE_DEVICES": lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),

    # 是否使用 HuggingFace 分词器
    "FD_USE_HF_TOKENIZER": lambda: bool(int(os.getenv("FD_USE_HF_TOKENIZER", "0"))),

    # 设置 ZMQ 初始化期间接收数据的高水位标记（HWM）
    "FD_ZMQ_SNDHWM": lambda: os.getenv("FD_ZMQ_SNDHWM", 0),

    # 缓存 KV 量化参数的目录
    "FD_CACHE_PARAMS": lambda: os.getenv("FD_CACHE_PARAMS", "none"),

    # 设置注意力机制后端，当前可设置为 "NATIVE_ATTN"、"APPEND_ATTN" 或 "MLA_ATTN"
    "FD_ATTENTION_BACKEND": lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),

    # 设置采样类别，当前可设置为 "base"、"base_non_truncated"、"air" 或 "rejection"
    "FD_SAMPLING_CLASS": lambda: os.getenv("FD_SAMPLING_CLASS", "base"),

    # 设置MoE后端，当前可设置为 "cutlass"、"marlin" 或 "triton"
    "FD_MOE_BACKEND": lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),

    # 是否使用 Machete 后端的 wint4 dense GEMM
    "FD_USE_MACHETE": lambda: os.getenv("FD_USE_MACHETE", "1"),

    # 是否在 KV cache 满时禁用重新计算请求
    "FD_DISABLED_RECOVER": lambda: os.getenv("FD_DISABLED_RECOVER", "0"),

    # 设置 Triton 内核 JIT 编译目录
    "FD_TRITON_KERNEL_CACHE_DIR": lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),

    # 是否从单机 PD 分离转换为集中式推理
    "FD_PD_CHANGEABLE": lambda: os.getenv("FD_PD_CHANGEABLE", "0"),

    # 是否使用DeepGemm后端的FP8 blockwise MoE
    "FD_USE_DEEP_GEMM": lambda: bool(int(os.getenv("FD_USE_DEEP_GEMM", "0"))),

    # 是否使用聚合发送
    "FD_USE_AGGREGATE_SEND": lambda: bool(int(os.getenv("FD_USE_AGGREGATE_SEND", "0"))),

    # 是否开启 Trace
    "TRACES_ENABLE": lambda: os.getenv("TRACES_ENABLE", "false"),

    # 设置 trace 服务名称
    "FD_SERVICE_NAME": lambda: os.getenv("FD_SERVICE_NAME", "FastDeploy"),

    # 设置 trace 主机名
    "FD_HOST_NAME": lambda: os.getenv("FD_HOST_NAME", "localhost"),

    # 设置 trace exporter
    "TRACES_EXPORTER": lambda: os.getenv("TRACES_EXPORTER", "console"),

    # 设置 trace exporter_otlp_endpoint
    "EXPORTER_OTLP_ENDPOINT": lambda: os.getenv("EXPORTER_OTLP_ENDPOINT"),

    # 设置 trace exporter_otlp_headers
    "EXPORTER_OTLP_HEADERS": lambda: os.getenv("EXPORTER_OTLP_HEADERS"),

    # 启用 kv cache block scheduler v1（不需要 kv_cache_ratio）
    "ENABLE_V1_KVCACHE_SCHEDULER": lambda: int(os.getenv("ENABLE_V1_KVCACHE_SCHEDULER", "1")),

    # 为 decoder 设置预分配 block 数量
    "FD_ENC_DEC_BLOCK_NUM": lambda: int(os.getenv("FD_ENC_DEC_BLOCK_NUM", "2")),

    # 启用单次执行步骤的最大 prefill
    "FD_ENABLE_MAX_PREFILL": lambda: int(os.getenv("FD_ENABLE_MAX_PREFILL", "0")),

    # 是否使用 PLUGINS
    "FD_PLUGINS": lambda: None if "FD_PLUGINS" not in os.environ else os.environ["FD_PLUGINS"].split(","),

    # 设置 trace 属性 job_id
    "FD_JOB_ID": lambda: os.getenv("FD_JOB_ID"),

    # 支持的最大连接数
    "FD_SUPPORT_MAX_CONNECTIONS": lambda: int(os.getenv("FD_SUPPORT_MAX_CONNECTIONS", "1024")),

    # Tensor Parallelism 组 GID 偏移量
    "FD_TP_GROUP_GID_OFFSET": lambda: int(os.getenv("FD_TP_GROUP_GID_OFFSET", "1000")),

    # 启用多 API 服务器
    "FD_ENABLE_MULTI_API_SERVER": lambda: bool(int(os.getenv("FD_ENABLE_MULTI_API_SERVER", "0"))),

    # 是否使用 Torch 模型格式
    "FD_FOR_TORCH_MODEL_FORMAT": lambda: bool(int(os.getenv("FD_FOR_TORCH_MODEL_FORMAT", "0"))),

    # 强制禁用默认的 chunked prefill
    "FD_DISABLE_CHUNKED_PREFILL": lambda: bool(int(os.getenv("FD_DISABLE_CHUNKED_PREFILL", "0"))),

    # 是否使用新的 get_output 和 save_output 方法 (0 或 1)
    "FD_USE_GET_SAVE_OUTPUT_V1": lambda: bool(int(os.getenv("FD_USE_GET_SAVE_OUTPUT_V1", "0"))),

    # 是否启用模型缓存功能
    "FD_ENABLE_MODEL_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_CACHE", "0"))),

    # 启用内部模块访问 LLMEngine
    "FD_ENABLE_INTERNAL_ADAPTER": lambda: int(os.getenv("FD_ENABLE_INTERNAL_ADAPTER", "0")),

    # LLMEngine 接收请求端口，在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ZMQ_RECV_REQUEST_SERVER_PORT": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORT", "8200"),

    # LLMEngine 发送响应端口，在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORT": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORT", "8201"),

    # LLMEngine 接收请求端口（多端口），在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ZMQ_RECV_REQUEST_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "8200"),

    # LLMEngine 发送响应端口（多端口），在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "8201"),

    # LLMEngine 接收控制命令端口，在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ZMQ_CONTROL_CMD_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_CONTROL_CMD_SERVER_PORTS", "8202"),

    # 是否启用 decode 缓存请求以预分配资源
    "FD_ENABLE_CACHE_TASK": lambda: os.getenv("FD_ENABLE_CACHE_TASK", "0"),

    # EP 中批处理 token 的超时时间
    "FD_EP_BATCHED_TOKEN_TIMEOUT": lambda: float(os.getenv("FD_EP_BATCHED_TOKEN_TIMEOUT", "0.1")),

    # PD 中最大预取请求数量
    "FD_EP_MAX_PREFETCH_TASK_NUM": lambda: int(os.getenv("FD_EP_MAX_PREFETCH_TASK_NUM", "8")),

    # 是否启用模型加载缓存。启用后，量化模型将作为缓存存储，以提高未来推理的加载效率
    "FD_ENABLE_MODEL_LOAD_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_LOAD_CACHE", "0"))),

    # 清除模型权重时是否清除 CPU 缓存
    "FD_ENABLE_SWAP_SPACE_CLEARING": lambda: int(os.getenv("FD_ENABLE_SWAP_SPACE_CLEARING", "0")),

    # 启用返回文本，在 FD_ENABLE_INTERNAL_ADAPTER=1 时使用
    "FD_ENABLE_RETURN_TEXT": lambda: bool(int(os.getenv("FD_ENABLE_RETURN_TEXT", "0"))),

    # 用于在模型推理思考时截断插入的字符串（ernie-45-vl 使用 </think>，ernie-x1 使用 \n</think>\n\n）
    "FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR": lambda: os.getenv("FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR", "</think>"),

    # cache_transfer_manager 进程残留时退出等待超时时间
    "FD_CACHE_PROC_EXIT_TIMEOUT": lambda: int(os.getenv("FD_CACHE_PROC_EXIT_TIMEOUT", "600")),

    # cache_transfer_manager 进程残留时连续错误阈值
    "FD_CACHE_PROC_ERROR_COUNT": lambda: int(os.getenv("FD_CACHE_PROC_ERROR_COUNT", "10")),

    # 服务认证所需的 API_KEY
    "FD_API_KEY": lambda: [] if "FD_API_KEY" not in os.environ else os.environ["FD_API_KEY"].split(","),

    # 多模态推理时存储特征的 BOS 的 AK
    "ENCODE_FEATURE_BOS_AK": lambda: os.getenv("ENCODE_FEATURE_BOS_AK"),

    # 多模态推理时存储特征的 BOS 的 SK
    "ENCODE_FEATURE_BOS_SK": lambda: os.getenv("ENCODE_FEATURE_BOS_SK"),

    # 多模态推理时存储特征的 BOS 的 ENDPOINT
    "ENCODE_FEATURE_ENDPOINT": lambda: os.getenv("ENCODE_FEATURE_ENDPOINT"),

    # 为 PD 分离启用离线性能测试模式
    "FD_OFFLINE_PERF_TEST_FOR_PD": lambda: int(os.getenv("FD_OFFLINE_PERF_TEST_FOR_PD", "0")),

    # 启用 E2W 张量转换
    "FD_ENABLE_E2W_TENSOR_CONVERT": lambda: int(os.getenv("FD_ENABLE_E2W_TENSOR_CONVERT", "0")),

    # 使用共享内存的引擎任务队列
    "FD_ENGINE_TASK_QUEUE_WITH_SHM": lambda: int(os.getenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")),

    # 填充位掩码批处理大小
    "FD_FILL_BITMASK_BATCH": lambda: int(os.getenv("FD_FILL_BITMASK_BATCH", "4")),

    # 启用 PDL
    "FD_ENABLE_PDL": lambda: int(os.getenv("FD_ENABLE_PDL", "1")),

    # 禁用 guidance 额外功能
    "FD_GUIDANCE_DISABLE_ADDITIONAL": lambda: bool(int(os.getenv("FD_GUIDANCE_DISABLE_ADDITIONAL", "1"))),

    # LLGuidance 日志级别
    "FD_LLGUIDANCE_LOG_LEVEL": lambda: int(os.getenv("FD_LLGUIDANCE_LOG_LEVEL", "0")),

    # HPU 上 MoE 计算处理的组中的 token 数量
    "FD_HPU_CHUNK_SIZE": lambda: int(os.getenv("FD_HPU_CHUNK_SIZE", "64")),

    # 在 HPU 上启用 FP8 校准
    "FD_HPU_MEASUREMENT_MODE": lambda: os.getenv("FD_HPU_MEASUREMENT_MODE", "0"),

    # Prefill 等待 decode 资源的秒数
    "FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS": lambda: int(os.getenv("FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS", "30")),

    # FMQ 配置 JSON
    "FMQ_CONFIG_JSON": lambda: os.getenv("FMQ_CONFIG_JSON", None),

    # OTLP Exporter 调度延迟（毫秒）
    "FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": lambda: int(os.getenv("FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "500")),

    # OTLP Exporter 最大导出批处理大小
    "FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": lambda: int(os.getenv("FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "64")),

    # Token 处理器健康检查超时时间
    "FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT": lambda: int(os.getenv("FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT", "120")),

    # XPU MoE FFN 量化类型映射
    "FD_XPU_MOE_FFN_QUANT_TYPE_MAP": lambda: os.getenv("FD_XPU_MOE_FFN_QUANT_TYPE_MAP", ""),

    # Worker 进程响应等待时的健康检查超时时间（秒），默认 30 秒
    "FD_WORKER_ALIVE_TIMEOUT": lambda: int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30")),

    # 控制是否收集用户信息，默认 0（收集）；1（不收集）
    "DO_NOT_TRACK" : lambda: (os.getenv("DO_NOT_TRACK", "0")) == "1",

    # 使用情况统计报告服务地址
    "FD_USAGE_STATS_SERVER": lambda: os.getenv(
        "FD_USAGE_STATS_SERVER", "http://fd-stats.baidu-int.com/fd/report/periodic"
    ),

    # 使用情况统计的来源信息，用户可主动设置
    "FD_USAGE_SOURCE": lambda: os.getenv("FD_USAGE_SOURCE", "Unknown"),

    # FastDeploy 配置根目录
    "FD_CONFIG_ROOT": lambda: os.path.expanduser(
        os.getenv("FD_CONFIG_ROOT", os.path.join(os.path.expanduser("~"), ".config", "fastdeploy"))
    ),
}
