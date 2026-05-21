# Intel HPU serving benchmark
These scripts are used to launch FastDeploy Paddle large model inference service for performance and stress testing.

## Main HPU-Specific Parameter
- `HPU_WARMUP_BUCKET`: Whether to enable warmup (1 means enabled)
- `HPU_WARMUP_MODEL_LEN`: Model length for warmup (including input and output)
- `MAX_PREFILL_NUM`: Maximum batch in prefill stage, default 3
- `BATCH_STEP_PREFILL`: Batch step in prefill stage, default 1
- `SEQUENCE_STEP_PREFILL`: Sequence step in prefill stage, default 128, same as block size
- `CONTEXT_BLOCK_STEP_PREFILL`: Step size for block hit when prefill caching is enabled, default 1
- `BATCH_STEP_DECODE`: Batch step in decode stage, default 4
- `BLOCK_STEP_DECODE`: Block step in decode stage, default 16
- `FLAGS_intel_hpu_recipe_cache_num`: Limit for HPU recipe cache number
- `FLAGS_intel_hpu_recipe_cache_config`: HPU recipe cache config, can be used for warmup optimization
- `GC_KERNEL_PATH`: The default path of the HPU TPC kernels library
- `HABANA_PROFILE`: Whether to enable profiler (1 means enabled)
- `PROFILE_START`: Profiler start step.
- `PROFILE_END`: Profiler end step.

## Usage
### 1. Start server
There are different setup scripts are provided to start the vllm server, one for RandomDataset and the other for ShareGPT.

Before running, please make sure to correctly set the model path and port number in the script.
```bash
./benchmark_paddle_hpu_server.sh
./benchmark_paddle_hpu_server_sharegpt.sh
```
You can use HPU_VISIBLE_DEVICES in the script to select the HPU card.

### 2. Run client
Correspondingly, there are different client test scripts. `benchmark_paddle_hpu_cli.sh` supports both variable and fixed length tests.

Before running, please make sure to correctly set the model path, port number, and input/output settings in the script.
```bash
./benchmark_paddle_hpu_cli.sh
./benchmark_paddle_hpu_cli_sharegpt.sh
```

### 3. Parse logs
After batch testing, run the following script to automatically parse the logs and generate a CSV file.
```python
python parse_benchmark_logs.py benchmark_fastdeploy_logs/[the targeted folder]
```
The performance data will be saved as a CSV file.

### 4. Analyse logs
During HPU_MODEL_RUNNER execution, performance logs are generated. The following script can parse these logs and produce performance graphs to help identify bottlenecks.
```python
python draw_benchmark_data.py benchmark_fastdeploy_logs/[the targeted folder]
```
The script will save the model execution times and batch tokens as a CSV file and plot them in a graph.

### 5. Accuracy test
Accuracy testing uses GSM8K. Use the following conversion to generate the test file.
```python
>>> import pandas as pd
>>> df = pd.read_parquet('tests/ci_validation/accuracy_cases/gsm8k.parquet', engine='pyarrow')
>>> df.to_json('test.jsonl', orient='records', lines=True)
```
Run the following command to perform the accuracy test.
```bash
python -u bench_gsm8k.py --port 8188 --num-questions 1319 --num-shots 5 --parallel 64
```

### 6. Offline demo
To run a offline demo on HPU quickly, after set model_path in offline_demo.py, run the start script directly.
```bash
./run_offline_demo.sh
```
