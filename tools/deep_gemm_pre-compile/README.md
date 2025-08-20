# DeepGEMM Pre-compilation Tool

This tool provides pre-compilation functionality for DeepGEMM kernels to optimize performance.

## Usage

### 1. Using Shell Script (Recommended)
```bash
bash pre_compile.sh \
    [MODEL_PATH] \
    [TP_SIZE] \
    [EP_SIZE] \
    [HAS_SHARED_EXPERTS] \
    [OUTPUT_FILE]
```

The script will:
1. Generate configurations
2. Pre-compile all kernels

### 2. Alternative: Manual Steps
If you need more control, you can run the steps manually:

#### Generate Configuration
```bash
python generate_config.py \
    --model /path/to/model \
    --tensor-parallel-size [TP_SIZE] \
    --expert-parallel-size [EP_SIZE] \
    --has-shared-experts [True/False] \
    --output [CONFIG_FILE]
```

Arguments:
- `--model`: Path to model directory containing config.json
- `--tensor-parallel-size`: Tensor parallel size (default: 1)
- `--expert-parallel-size`: Expert parallel size (default: 8)
- `--has-shared-experts`: Whether model has shared experts (default: False)
- `--output`: Output config file path (default: ./deep_gemm_pre_compile_config.jsonl)

#### Pre-compile Kernels
```bash
python pre_compile.py \
    --config-file [CONFIG_FILE] \
    --expert-parallel-size [EP_SIZE] \
    --num-threads [NUM_THREADS]
```

Arguments:
- `--config-file`: Path to config file generated in step 1
- `--expert-parallel-size`: Expert parallel size (must match step 1)
- `--num-threads`: Number of compilation threads (default: CPU cores)

## Environment Variables
- `PRE_COMPILE_LOG_LEVEL`: Set log level (DEBUG/INFO/WARNING/ERROR)
- `DG_CACHE_DIR`: Cache directory for compiled kernels (default: ./deep_gemm_cache)

## Notes
- For best performance, set `--num-threads` to the number of available CPU cores
- The compilation process may take significant time depending on configuration size
- Compiled kernels will be cached in `DG_CACHE_DIR`
