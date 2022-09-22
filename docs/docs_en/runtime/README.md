# FastDeploy Inference Backend

FastDeploy currently integrates with a wide range of inference backends. The following table summarises these integrated backends and information, including the platforms and hardware.

| Inference Backend | Platform                        | Hardware | Supported Model Format |
|:----------------- |:------------------------------- |:-------- |:---------------------- |
| Paddle Inference  | Windows(x64)/Linux(x64)         | GPU/CPU  | Paddle                 |
| ONNX Runtime      | Windows(x64)/Linux(x64/aarch64) | GPU/CPU  | Paddle/ONNX            |
| TensorRT          | Windows(x64)/Linux(x64/jetson)  | GPU      | Paddle/ONNX            |
| OpenVINO          | Windows(x64)/Linux(x64)         | CPU      | Paddle/ONNX            |
| Poros[Incoming]   | Linux(x64)                      | CPU/GPU  | TorchScript            |

Backends in FastDeploy are independent and developers can choose to enable one or more of them for customized compilation.
The `Runtime` module in FastDeploy provides a unified API for all backends. See the [FastDeploy Runtime User Guideline](usage.md) for more details.

## Related Files

- [FastDeploy Compile](../compile)
