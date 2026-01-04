[简体中文](../zh/usage/environment_variables.md)

# FastDeploy Environment Variables

FastDeploy's environment variables are defined in `fastdeploy/envs.py` at the root of the repository. Below is the documentation:

```python
environment_variables: dict[str, Callable[[], Any]] = {
    # Whether to use BF16 on CPU
    "FD_CPU_USE_BF16": lambda: os.getenv("FD_CPU_USE_BF16", "False"),

    # CUDA architecture versions used when building FastDeploy (string list, e.g. [80,90])
    "FD_BUILDING_ARCS": lambda: os.getenv("FD_BUILDING_ARCS", "[]"),

    # Log directory
    "FD_LOG_DIR": lambda: os.getenv("FD_LOG_DIR", "log"),

    # Enable debug mode (0 or 1)
    "FD_DEBUG": lambda: int(os.getenv("FD_DEBUG", "0")),

    # FastDeploy log retention days
    "FD_LOG_BACKUP_COUNT": lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),

    # Model download source, can be "AISTUDIO", "MODELSCOPE" or "HUGGINGFACE"
    "FD_MODEL_SOURCE": lambda: os.getenv("FD_MODEL_SOURCE", "AISTUDIO"),

    # Model download cache directory
    "FD_MODEL_CACHE": lambda: os.getenv("FD_MODEL_CACHE", None),

    # Maximum number of stop sequences
    "FD_MAX_STOP_SEQS_NUM": lambda: int(os.getenv("FD_MAX_STOP_SEQS_NUM", "5")),

    # Maximum length of stop sequences
    "FD_STOP_SEQS_MAX_LEN": lambda: int(os.getenv("FD_STOP_SEQS_MAX_LEN", "8")),

    # GPU devices to use (comma-separated string, e.g. 0,1,2)
    "CUDA_VISIBLE_DEVICES": lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),

    # Whether to use HuggingFace tokenizer (0 or 1)
    "FD_USE_HF_TOKENIZER": lambda: bool(int(os.getenv("FD_USE_HF_TOKENIZER", "0"))),

    # ZMQ send high-water mark (HWM) during initialization
    "FD_ZMQ_SNDHWM": lambda: os.getenv("FD_ZMQ_SNDHWM", 0),

    # Directory for caching KV quantization parameters
    "FD_CACHE_PARAMS": lambda: os.getenv("FD_CACHE_PARAMS", "none"),

    # Attention backend ("NATIVE_ATTN", "APPEND_ATTN", or "MLA_ATTN")
    "FD_ATTENTION_BACKEND": lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),

    # Sampling class ("base", "base_non_truncated", "air", or "rejection")
    "FD_SAMPLING_CLASS": lambda: os.getenv("FD_SAMPLING_CLASS", "base"),

    # MoE backend ("cutlass", "marlin", or "triton")
    "FD_MOE_BACKEND": lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),

    # Whether to use Machete for wint4 dense GEMM
    "FD_USE_MACHETE": lambda: os.getenv("FD_USE_MACHETE", "1"),

    # Whether to disable recompute the request when the KV cache is full
    "FD_DISABLED_RECOVER": lambda: os.getenv("FD_DISABLED_RECOVER", "0"),

    # Triton kernel JIT compilation directory
    "FD_TRITON_KERNEL_CACHE_DIR": lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),

    # Switch from standalone PD to centralized inference (0 or 1)
    "FD_PD_CHANGEABLE": lambda: os.getenv("FD_PD_CHANGEABLE", "0"),

    # Whether to use DeepGemm for FP8 blockwise MoE
    "FD_USE_DEEP_GEMM": lambda: bool(int(os.getenv("FD_USE_DEEP_GEMM", "0"))),

    # Whether to use aggregate send
    "FD_USE_AGGREGATE_SEND": lambda: bool(int(os.getenv("FD_USE_AGGREGATE_SEND", "0"))),

    # Whether to open Trace
    "TRACES_ENABLE": lambda: os.getenv("TRACES_ENABLE", "false"),

    # Set trace server name
    "FD_SERVICE_NAME": lambda: os.getenv("FD_SERVICE_NAME", "FastDeploy"),

    # Set trace host name
    "FD_HOST_NAME": lambda: os.getenv("FD_HOST_NAME", "localhost"),

    # Set trace exporter
    "TRACES_EXPORTER": lambda: os.getenv("TRACES_EXPORTER", "console"),

    # Set trace exporter_otlp_endpoint
    "EXPORTER_OTLP_ENDPOINT": lambda: os.getenv("EXPORTER_OTLP_ENDPOINT"),

    # Set trace exporter_otlp_headers
    "EXPORTER_OTLP_HEADERS": lambda: os.getenv("EXPORTER_OTLP_HEADERS"),

    # Enable kv cache block scheduler v1 (no need for kv_cache_ratio)
    "ENABLE_V1_KVCACHE_SCHEDULER": lambda: int(os.getenv("ENABLE_V1_KVCACHE_SCHEDULER", "1")),

    # Set prealloc block num for decoder
    "FD_ENC_DEC_BLOCK_NUM": lambda: int(os.getenv("FD_ENC_DEC_BLOCK_NUM", "2")),

    # Enable max prefill of one execute step
    "FD_ENABLE_MAX_PREFILL": lambda: int(os.getenv("FD_ENABLE_MAX_PREFILL", "0")),

    # Whether to use PLUGINS
    "FD_PLUGINS": lambda: None if "FD_PLUGINS" not in os.environ else os.environ["FD_PLUGINS"].split(","),

    # Set trace attribute job_id
    "FD_JOB_ID": lambda: os.getenv("FD_JOB_ID"),

    # Support max connections
    "FD_SUPPORT_MAX_CONNECTIONS": lambda: int(os.getenv("FD_SUPPORT_MAX_CONNECTIONS", "1024")),

    # Offset for Tensor Parallelism group GID
    "FD_TP_GROUP_GID_OFFSET": lambda: int(os.getenv("FD_TP_GROUP_GID_OFFSET", "1000")),

    # Enable multi api server
    "FD_ENABLE_MULTI_API_SERVER": lambda: bool(int(os.getenv("FD_ENABLE_MULTI_API_SERVER", "0"))),

    # Whether to use Torch model format
    "FD_FOR_TORCH_MODEL_FORMAT": lambda: bool(int(os.getenv("FD_FOR_TORCH_MODEL_FORMAT", "0"))),

    # Force disable default chunked prefill
    "FD_DISABLE_CHUNKED_PREFILL": lambda: bool(int(os.getenv("FD_DISABLE_CHUNKED_PREFILL", "0"))),

    # Whether to use new get_output and save_output method (0 or 1)
    "FD_USE_GET_SAVE_OUTPUT_V1": lambda: bool(int(os.getenv("FD_USE_GET_SAVE_OUTPUT_V1", "0"))),

    # Whether to enable model cache feature
    "FD_ENABLE_MODEL_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_CACHE", "0"))),

    # Enable internal module to access LLMEngine
    "FD_ENABLE_INTERNAL_ADAPTER": lambda: int(os.getenv("FD_ENABLE_INTERNAL_ADAPTER", "0")),

    # LLMEngine receive requests port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORT": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORT", "8200"),

    # LLMEngine send response port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORT": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORT", "8201"),

    # LLMEngine receive requests port (multiple ports), used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_RECV_REQUEST_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_RECV_REQUEST_SERVER_PORTS", "8200"),

    # LLMEngine send response port (multiple ports), used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_SEND_RESPONSE_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_SEND_RESPONSE_SERVER_PORTS", "8201"),

    # LLMEngine receive control command port, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ZMQ_CONTROL_CMD_SERVER_PORTS": lambda: os.getenv("FD_ZMQ_CONTROL_CMD_SERVER_PORTS", "8202"),

    # Whether to enable the decode caches requests for preallocating resource
    "FD_ENABLE_CACHE_TASK": lambda: os.getenv("FD_ENABLE_CACHE_TASK", "0"),

    # Batched token timeout in EP
    "FD_EP_BATCHED_TOKEN_TIMEOUT": lambda: float(os.getenv("FD_EP_BATCHED_TOKEN_TIMEOUT", "0.1")),

    # Max pre-fetch requests number in PD
    "FD_EP_MAX_PREFETCH_TASK_NUM": lambda: int(os.getenv("FD_EP_MAX_PREFETCH_TASK_NUM", "8")),

    # Enable or disable model caching. When enabled, the quantized model is stored as a cache for future inference to improve loading efficiency
    "FD_ENABLE_MODEL_LOAD_CACHE": lambda: bool(int(os.getenv("FD_ENABLE_MODEL_LOAD_CACHE", "0"))),

    # Whether to clear cpu cache when clearing model weights
    "FD_ENABLE_SWAP_SPACE_CLEARING": lambda: int(os.getenv("FD_ENABLE_SWAP_SPACE_CLEARING", "0")),

    # Enable return text, used when FD_ENABLE_INTERNAL_ADAPTER=1
    "FD_ENABLE_RETURN_TEXT": lambda: bool(int(os.getenv("FD_ENABLE_RETURN_TEXT", "0"))),

    # Used to truncate the string inserted during thinking when reasoning in a model. (</think> for ernie-45-vl, \n</think>\n\n for ernie-x1)
    "FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR": lambda: os.getenv("FD_LIMIT_THINKING_CONTENT_TRUNCATE_STR", "</think>"),

    # Timeout for cache_transfer_manager process exit
    "FD_CACHE_PROC_EXIT_TIMEOUT": lambda: int(os.getenv("FD_CACHE_PROC_EXIT_TIMEOUT", "600")),

    # Count for cache_transfer_manager process error
    "FD_CACHE_PROC_ERROR_COUNT": lambda: int(os.getenv("FD_CACHE_PROC_ERROR_COUNT", "10")),

    # API_KEY required for service authentication
    "FD_API_KEY": lambda: [] if "FD_API_KEY" not in os.environ else os.environ["FD_API_KEY"].split(","),

    # The AK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_AK": lambda: os.getenv("ENCODE_FEATURE_BOS_AK"),

    # The SK of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_BOS_SK": lambda: os.getenv("ENCODE_FEATURE_BOS_SK"),

    # The ENDPOINT of bos storing the features while multi_modal infer
    "ENCODE_FEATURE_ENDPOINT": lambda: os.getenv("ENCODE_FEATURE_ENDPOINT"),

    # Enable offline perf test mode for PD disaggregation
    "FD_OFFLINE_PERF_TEST_FOR_PD": lambda: int(os.getenv("FD_OFFLINE_PERF_TEST_FOR_PD", "0")),

    # Enable E2W tensor convert
    "FD_ENABLE_E2W_TENSOR_CONVERT": lambda: int(os.getenv("FD_ENABLE_E2W_TENSOR_CONVERT", "0")),

    # Engine task queue with shared memory
    "FD_ENGINE_TASK_QUEUE_WITH_SHM": lambda: int(os.getenv("FD_ENGINE_TASK_QUEUE_WITH_SHM", "0")),

    # Fill bitmask batch size
    "FD_FILL_BITMASK_BATCH": lambda: int(os.getenv("FD_FILL_BITMASK_BATCH", "4")),

    # Enable PDL
    "FD_ENABLE_PDL": lambda: int(os.getenv("FD_ENABLE_PDL", "1")),

    # Disable guidance additional feature
    "FD_GUIDANCE_DISABLE_ADDITIONAL": lambda: bool(int(os.getenv("FD_GUIDANCE_DISABLE_ADDITIONAL", "1"))),

    # LLGuidance log level
    "FD_LLGUIDANCE_LOG_LEVEL": lambda: int(os.getenv("FD_LLGUIDANCE_LOG_LEVEL", "0")),

    # Number of tokens in the group for Mixture of Experts (MoE) computation processing on HPU
    "FD_HPU_CHUNK_SIZE": lambda: int(os.getenv("FD_HPU_CHUNK_SIZE", "64")),

    # Enable FP8 calibration on HPU
    "FD_HPU_MEASUREMENT_MODE": lambda: os.getenv("FD_HPU_MEASUREMENT_MODE", "0"),

    # Prefill wait decode resource seconds
    "FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS": lambda: int(os.getenv("FD_PREFILL_WAIT_DECODE_RESOURCE_SECONDS", "30")),

    # FMQ config JSON
    "FMQ_CONFIG_JSON": lambda: os.getenv("FMQ_CONFIG_JSON", None),

    # OTLP Exporter schedule delay millis
    "FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": lambda: int(os.getenv("FD_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "500")),

    # OTLP Exporter max export batch size
    "FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": lambda: int(os.getenv("FD_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "64")),

    # Token processor health timeout
    "FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT": lambda: int(os.getenv("FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT", "120")),

    # XPU MoE FFN quant type map
    "FD_XPU_MOE_FFN_QUANT_TYPE_MAP": lambda: os.getenv("FD_XPU_MOE_FFN_QUANT_TYPE_MAP", ""),

    # Worker process health check timeout when waiting for responses in seconds (default: 30)
    "FD_WORKER_ALIVE_TIMEOUT": lambda: int(os.getenv("FD_WORKER_ALIVE_TIMEOUT", "30")),
}
```
