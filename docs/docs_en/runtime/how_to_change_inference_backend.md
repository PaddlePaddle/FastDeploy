# How to change inference backend

Vision models in FastDeploy support a wide range of backends, including

- OpenVINO (support models in Paddle/ONNX formats and inference on CPU only)
- ONNX Runtime (support models in Paddle/ONNX formats and inference on CPU or GPU）
- TensorRT (Support models in Paddle/ONNX formats and inference on GPU only
- Paddle Inference(support models in Paddle format and inference on CPU or GPU)

All the models change its inference backend through RuntimeOption

**Python**

```
import fastdeploy as fd
option = fd.RuntimeOption()

# Change inference on CPU/GPU
option.use_cpu()
option.use_gpu()

# Change backend
option.use_paddle_backend() # Paddle Inference
option.use_trt_backend() # TensorRT
option.use_openvino_backend() # OpenVINO
option.use_ort_backend() # ONNX Runtime
```

**C++**

```
fastdeploy::RuntimeOption option;

# Change inference on CPU/GPU
option.UseCpu();
option.UseGpu();

# Change backend
option.UsePaddleBackend(); // Paddle Inference
option.UseTrtBackend(); // TensorRT
option.UseOpenVINOBackend(); // OpenVINO
option.UseOrtBackend(); // ONNX Runtime
```

Please refer to `FastDeploy/examples/vision` for python or c++ inference code of different models.

For more deployment methods of `RuntimeOption`, please refer to [RuntimeOption API](../../docs/api/runtime/runtime_option.md)
