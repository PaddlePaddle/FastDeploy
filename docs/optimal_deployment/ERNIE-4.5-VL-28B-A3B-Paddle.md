
# ERNIE-4.5-VL-28B-A3B-Paddle

**Note**: To enable multi-modal support, add the `--enable-mm` flag to your configuration.

## Performance Optimization Guide

To help you achieve the **best performance** with our model, here are several important parameters you may want to adjust. Please read through the following recommendations and tips:

###  **Context Length**  
- **Parameter**: `--max-model-len`  
- **Description**: Controls the maximum context length the model can process.
- **Recommendation**: We suggest setting this to **32k tokens** for balanced performance and memory usage.
- **Advanced**: If your hardware allows and you need even longer contexts, you can set it up to **128k tokens**.
 
   ⚠️ Note: Longer contexts require significantly more GPU memory. Please ensure your hardware is sufficient before increasing this value.

### **Multi-Image & Multi-Video Input**  
- **Parameter**: `--limit-mm-per-prompt`  
- **Description**: Our model supports multiple images and videos per prompt. Use this parameter to limit the number of images and videos per request, ensuring efficient resource utilization. 
- **Recommendation**: We suggest setting this to **100 images** and **100 videos** per prompt for balanced performance and memory usage.

### **Optimization Recommendations**
> **chunked prefill**
- **Parameter**: `--enable-chunked-prefill`
- **Why enable?**
              
   Enabling chunked prefill can **reduce peak memory usage** and **increase inference speed**.
- **Additional options**:
   - --max-num-batched-tokens
   - --max-num-partial-prefills
   - --max-long-partial-prefills
- **Tip**:

    The detailed workings of these auxiliary parameters are complex—feel free to use the example values provided in our documentation or scripts.

> **prefix caching**

⚠️ Note: Prefix caching is currently not supported in multi-modal mode.

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

| Devices | Runnable Quantization | TPS(t/s) |  Latency(ms) |
|:----------:|:----------:|:------:|:------:|
| A30 | wint4 | 432.99 | 17396.92 |
| L20 | wint4<br>wint8 | 3311.34<br>2423.36  | 46566.81<br>60790.91 |
| H20 | wint4<br>wint8<br>bfloat16 | 3827.27<br>3578.23<br>4100.83  | 89770.14<br>95434.02<br>84543.00  |
| A100| wint4<br>wint8<br>bfloat16 | 4970.15<br>4842.86<br>3946.32 | 68316.08<br>78518.78<br>87448.57 |
| H800| wint4<br>wint8<br>bfloat16 | 7450.01<br>7455.76<br>6351.90 | 49076.18<br>49253.59<br>54309.99 |

> Devices not verified can still run if their RAM/VRAM meets the requirements.


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
       --kv-cache-ratio 0.8 \
       --enable-chunked-prefill \
       --max-num-batched-tokens 1024 \
       --max-num-partial-prefills 3 \
       --max-long-partial-prefills 3 \
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
       --kv-cache-ratio 0.8 \
       --enable-chunked-prefill \
       --max-num-batched-tokens 1024 \
       --max-num-partial-prefills 3 \
       --max-long-partial-prefills 3 \
       --quantization wint8 \
       --enable-mm \
```