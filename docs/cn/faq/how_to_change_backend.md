
[English](../../en/faq/how_to_change_backend.md) | 中文

# 如何切换模型推理后端

FastDeploy中各视觉模型可支持多种后端，包括
- OpenVINO (支持Paddle/ONNX两种格式模型, 仅支持CPU上推理)
- ONNX Runtime (支持Paddle/ONNX两种格式模型， 支持CPU/GPU）
- TensorRT (支持Paddle/ONNX两种格式模型，仅支持GPU上推理)
- Paddle Inference(支持Paddle格式模型， 支持CPU/GPU)

所有模型切换后端方式均通过RuntimeOption进行切换，

**Python**
```python
import fastdeploy as fd
option = fd.RuntimeOption()

# 切换使用CPU/GPU
option.use_cpu()
option.use_gpu()

# 切换不同后端
option.use_paddle_backend() # Paddle Inference
option.use_trt_backend() # TensorRT
option.use_openvino_backend() # OpenVINO
option.use_ort_backend() # ONNX Runtime

```

**C++**
```C++
fastdeploy::RuntimeOption option;

// 切换使用CPU/GPU
option.UseCpu();
option.UseGpu();

// 切换不同后端
option.UsePaddleBackend(); // Paddle Inference
option.UseTrtBackend(); // TensorRT
option.UseOpenVINOBackend(); // OpenVINO
option.UseOrtBackend(); // ONNX Runtime
```

具体示例可参阅`FastDeploy/examples/vision`下不同模型的python或c++推理代码

更多`RuntimeOption`的配置方式查阅FastDeploy API文档
- [Python API]()
- [C++ API]()
