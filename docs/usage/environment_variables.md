# FastDeploy Environment Variables

FastDeploy's environment variables are defined in `fastdeploy/envs.py` at the root of the repository. Below is the documentation:

```python
environment_variables: dict[str, Callable[[], Any]] = {
    # CUDA architecture versions used when building FastDeploy (string list, e.g. [80,90])
    "FD_BUILDING_ARCS":
    lambda: os.getenv("FD_BUILDING_ARCS", "[]"),

    # Log directory
    "FD_LOG_DIR":
    lambda: os.getenv("FD_LOG_DIR", "log"),

    # Enable debug mode (0 or 1)
    "FD_DEBUG":
    lambda: os.getenv("FD_DEBUG", "0"),

    # FastDeploy log retention days
    "FD_LOG_BACKUP_COUNT":
    lambda: os.getenv("FD_LOG_BACKUP_COUNT", "7"),

    # Model download cache directory
    "FD_MODEL_CACHE":
    lambda: os.getenv("FD_MODEL_CACHE", None),

    # Maximum number of stop sequences
    "FD_MAX_STOP_SEQS_NUM":
    lambda: os.getenv("FD_MAX_STOP_SEQS_NUM", "5"),

    # Maximum length of stop sequences
    "FD_STOP_SEQS_MAX_LEN":
    lambda: os.getenv("FD_STOP_SEQS_MAX_LEN", "8"),

    # GPU devices to use (comma-separated string, e.g. 0,1,2)
    "CUDA_VISIBLE_DEVICES":
    lambda: os.getenv("CUDA_VISIBLE_DEVICES", None),

    # Whether to use HuggingFace tokenizer (0 or 1)
    "FD_USE_HF_TOKENIZER":
    lambda: os.getenv("FD_USE_HF_TOKENIZER", 0),

    # ZMQ send high-water mark (HWM) during initialization
    "FD_ZMQ_SNDHWM":
    lambda: os.getenv("FD_ZMQ_SNDHWM", 10000),

    # Directory for caching KV quantization parameters
    "FD_CACHE_PARAMS":
    lambda: os.getenv("FD_CACHE_PARAMS", "none"),

    # Attention backend ("NATIVE_ATTN", "APPEND_ATTN", or "MLA_ATTN")
    "FD_ATTENTION_BACKEND":
    lambda: os.getenv("FD_ATTENTION_BACKEND", "APPEND_ATTN"),

    # Sampling class ("base", "air", or "rejection")
    "FD_SAMPLING_CLASS":
    lambda: os.getenv("FD_SAMPLING_CLASS", "base"),

    # MoE backend ("cutlass", "marlin", or "triton")
    "FD_MOE_BACKEND":
    lambda: os.getenv("FD_MOE_BACKEND", "cutlass"),

    # Triton kernel JIT compilation directory
    "FD_TRITON_KERNEL_CACHE_DIR":
    lambda: os.getenv("FD_TRITON_KERNEL_CACHE_DIR", None),

    # Switch from standalone PD to centralized inference (0 or 1)
    "FD_PD_CHANGEABLE":
    lambda: os.getenv("FD_PD_CHANGEABLE", "1"),
  
}
```