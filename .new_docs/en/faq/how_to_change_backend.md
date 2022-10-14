# How to Change Model Inference Backend

FastDeploy supports various backends, including

- OpenVINO (supports Paddle/ONNX formats, CPU inference only )
- ONNX Runtime (supports Paddle/ONNX formats, inference on CPU/GPU)
- TensorRT (supports Paddle/ONNX formats, GPU inference only）
- Paddle Inference (supports Paddle format, inference on CPU/GPU)

All models can backend via RuntimeOption


**Python**
```python
import fastdeploy as fd
option = fd.RuntimeOption()

# Change CPU/GPU
option.use_cpu()
option.use_gpu()

# Change the Backend
option.use_paddle_backend() # Paddle Inference
option.use_trt_backend() # TensorRT
option.use_openvino_backend() # OpenVINO
option.use_ort_backend() # ONNX Runtime

```

**C++**
```C++
fastdeploy::RuntimeOption option;

// Change CPU/GPU
option.UseCpu();
option.UseGpu();

// Change the Backend
option.UsePaddleBackend(); // Paddle Inference
option.UseTrtBackend(); // TensorRT
option.UseOpenVINOBackend(); // OpenVINO
option.UseOrtBackend(); // ONNX Runtime
```

具体示例可参阅`FastDeploy/examples/vision`下不同模型的python或c++推理代码

For more deployment methods, please refer to FastDeploy API tutorials. 

- [Python API]()
- [C++ API]()
