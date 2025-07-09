
# ERNIE-4.5-VL-28B-A3B-Paddle

**Note**: To enable multi-modal support, add the `--enable-mm` flag to your configuration.

## Performance Optimization Guide

To help you achieve the **best performance** with our model, here are several important parameters you may want to adjust. Please read through the following recommendations and tips:

###  **Context Length**  
- **Parameter**: `--max-model-len`  
- **Description**: Controls the maximum context length the model can process.
- **Recommendation**: We suggest setting this to **32k tokens** (32768) for balanced performance and memory usage.
- **Advanced**: If your hardware allows and you need even longer contexts, you can set it up to **128k tokens** (131072).
 
   ⚠️ Note: Longer contexts require significantly more GPU memory. Please ensure your hardware is sufficient before increasing this value.

###  **Maximum Sequence Number**  
- **Parameter**: `--max-num-seqs`  
- **Description**: Controls the maximum number of sequences the service can handle, supports 1~256.
- **Recommendation**: If you don't know the average number of sequences in your actual application scenario, we recommend setting it to **256**. If the average number of sequences in your application scenario is significantly less than 256, we recommend setting it to a slightly higher value than the average to further optimize memory usage and service performance.

### **Multi-Image & Multi-Video Input**  
- **Parameter**: `--limit-mm-per-prompt`  
- **Description**: Our model supports multiple images and videos per prompt. Use this parameter to limit the number of images and videos per request, ensuring efficient resource utilization. 
- **Recommendation**: We suggest setting this to **100 images** and **100 videos** per prompt for balanced performance and memory usage.

### **Optimization Recommendations**
> **chunked prefill**
- **Parameter**: `--enable-chunked-prefill`
- **Why enable?**
              
   Enabling chunked prefill can **reduce peak memory usage** and **increase throughput**.
- **Additional options**:
   - `--max-num-batched-tokens`: Limit the maximum token count per chunk, with a recommended setting of 1,024.

> **prefix caching**

⚠️ Prefix caching is currently not supported in multi-modal mode.

### **Quantization Precision**:
- **Parameter**: `--quantization`

- **Supported Types**:
  - wint4 (recommended for most users)
  - wint8
  - Default: bfloat16 (if no quantization parameter is set)

- **Recommendation**:
Unless you have extremely strict precision requirements, we strongly recommend using wint4 quantization. This will dramatically reduce memory footprint and improve throughput.
If you need slightly higher precision, try wint8.
Only use bfloat16 if your use case demands the highest possible accuracy, as it requires much more memory.

- **Verified Devices and Performance**

| Devices | Runnable Quantization | TPS(tok/s) |  Latency(ms) |
|:----------:|:----------:|:------:|:------:|
| A30 | wint4 | 432.99 | 17396.92 |
| L20 | wint4<br>wint8 | 3311.34<br>2423.36  | 46566.81<br>60790.91 |
| H20 | wint4<br>wint8<br>bfloat16 | 3827.27<br>3578.23<br>4100.83  | 89770.14<br>95434.02<br>84543.00  |
| A100| wint4<br>wint8<br>bfloat16 | 4970.15<br>4842.86<br>3946.32 | 68316.08<br>78518.78<br>87448.57 |
| H800| wint4<br>wint8<br>bfloat16 | 7450.01<br>7455.76<br>6351.90 | 49076.18<br>49253.59<br>54309.99 |

> ⚠️ Note: Devices not verified can still run if their CPU/GPU memory meets the requirements.

### **Other Configurations**
> **gpu-memory-utilization**
- **Parameter**: `--gpu-memory-utilization`
- **Usage**: Controls the available GPU memory allocated for FastDeploy service initialization, with a default value of 0.9 (reserving 10% of GPU memory as buffer).
- **Recommendation**: It is recommended to set it to 0.9 on Nvidia Ampere GPUs, and to 0.8–0.9 on Hopper GPUs. If you encounter an out-of-memory error during service stress testing, you can try lowering this value.

> **kv-cache-ratio**
- **Parameter**: `--kv-cache-ratio`
- **Usage**: It is used to control the allocation ratio of GPU memory for the kv cache. The default value is 0.75, meaning that 75% of the kv cache memory is allocated to the input.
- **Recommendation**: Theoretically, the optimal value should be set to $\frac{average\ input\ length}{average\ input\ length+average\ output\ length}$ for your application scenario. If you are unsure, you can keep the default value.
                          
### **Example**: Single-card wint4 with 32K context length 
```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --tensor-parallel-size 1 \
       --max-model-len 32768 \
       --max-num-seqs 256 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl \
       --gpu-memory-utilization 0.9 \
       --kv-cache-ratio 0.75 \
       --enable-chunked-prefill \
       --max-num-batched-tokens 1024 \
       --quantization wint4 \
       --enable-mm \
```
###  **Example**: Dual-GPU Wint8 with 128K Context Length Configuration 
```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model baidu/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --tensor-parallel-size 2 \
       --max-model-len 131072 \
       --max-num-seqs 256 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl \
       --gpu-memory-utilization 0.9 \
       --kv-cache-ratio 0.75 \
       --enable-chunked-prefill \
       --max-num-batched-tokens 1024 \
       --quantization wint8 \
       --enable-mm \
```