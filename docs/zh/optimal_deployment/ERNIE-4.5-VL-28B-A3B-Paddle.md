
# ERNIE-4.5-VL-28B-A3B-Paddle

**注意：** 使用多模服务部署需要在配置中添加参数 `--enable-mm`。

## 性能优化指南

为了帮助您在使用本模型时达到**最佳性能**，以下是一些需要调整参数以及我们的建议。请仔细阅读以下推荐和小贴士：

###  **上下文长度**  
- **参数：** `--max-model-len`  
- **描述：** 控制模型可处理的最大上下文长度。
- **推荐：** 考虑到性能和内存占用的平衡，我们建议设置为**32k**（32768）。
- **进阶：** 如果您的硬件允许，并且您需要更长的上下文长度，我们也支持**128k**（131072）长度的上下文。
 
   ⚠️ 注：更长的上下文会显著增加GPU显存需求，设置更长的上下文之前确保硬件资源是满足的。

###  **最大序列数量**  
- **参数：** `--max-num-seqs`  
- **描述：** 控制服务可以处理的最大序列数量，支持1～256。
- **推荐：** 如果您不知道实际应用场景中请求的平均序列数量是多少，我们建议设置为**256**。如果您的应用场景中请求的平均序列数量明显少于256，我们建议设置为一个略大于平均值的较小值，以进一步优化显存占用以优化服务性能。

### **多图、多视频输入**  
- **参数**：`--limit-mm-per-prompt`  
- **描述**：我们的模型支持单次提示词（prompt）中输入多张图片和视频。请使用此参数限制每次请求的图片/视频数量，以确保资源高效利用。
- **推荐**：我们建议将单次提示词（prompt）中的图片和视频数量均设置为100个，以平衡性能与内存占用。

### **性能调优**
> **chunked prefill**
- **参数：**：`--enable-chunked-prefill`
- **用处：**
              
   开启 `chunked prefill` 可**降低显存峰值**并**提升服务吞吐**。

- **其他配置**:
   - `--max-num-batched-tokens`：限制每个chunk的最大token数量，推荐设置为1024。

> **上下文缓存**

⚠️ 目前上下缓存的功能在多模态上还未支持。

### **量化精度**
- **参数：** `--quantization`

- **已支持的精度类型：**
  - wint4 (适合大多数用户)
  - wint8
  - bfloat16 (未设置 `--quantization` 参数时，默认使用bfloat16)

- **推荐：** 
    - 除非您有极其严格的精度要求，否则我们强烈建议使用wint4量化。这将显著降低内存占用并提升吞吐量。
    - 若需要稍高的精度，可尝试wint8。
    - 仅当您的应用场景对精度有极致要求时候才尝试使用bfloat16，因为它需要更多显存。

- **验证过的设备和部分测试数据：**

    | 设备 | 可运行的量化精度 | TPS(tok/s) |  Latency(ms) |
    |:----------:|:----------:|:------:|:------:|
    | A30 | wint4 | 432.99 | 17396.92 |
    | L20 | wint4<br>wint8 | 3311.34<br>2423.36  | 46566.81<br>60790.91 |
    | H20 | wint4<br>wint8<br>bfloat16 | 3827.27<br>3578.23<br>4100.83  | 89770.14<br>95434.02<br>84543.00  |
    | A100| wint4<br>wint8<br>bfloat16 | 4970.15<br>4842.86<br>3946.32 | 68316.08<br>78518.78<br>87448.57 |
    | H800| wint4<br>wint8<br>bfloat16 | 7450.01<br>7455.76<br>6351.90 | 49076.18<br>49253.59<br>54309.99 |

> ⚠️ 注：没有验证过的设备和精度组合，只要内存和显存满足要求也是可以运行的。

### **其他配置**
> **gpu-memory-utilization**
- **参数：** `--gpu-memory-utilization`
- **用处：** 用于控制 FastDeploy 初始化服务的可用显存，默认0.9，即预留10%的显存备用。
- **推荐：** A卡上推荐0.9，H卡上推荐0.8～0.9。如果服务压测时提示显存不足，可以尝试调低该值。

> **kv-cache-ratio**
- **参数：**`--kv-cache-ratio`
- **用处：** 用于控制 kv cache 显存的分配比例，默认0.75，即75%的 kv cache 显存给输入。
- **推荐：** 理论最佳值应设置为应用场景的$\frac{平均输入长度}{平均输入长度+平均输出长度}$，如果您无法确定，保持默认值即可。
                          
### **示例：** 单卡、wint4、32K上下文部署命令
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
###  **示例：** 双卡、wint8、128K上下文部署命令
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