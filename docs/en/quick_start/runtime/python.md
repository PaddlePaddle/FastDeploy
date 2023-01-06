English | [中文](../../../cn/quick_start/runtime/python.md)
# Python Inference

Please check out the FastDeploy is already installed in your environment. You can refer to [FastDeploy Installation](../../build_and_install/) to install the pre-compiled FastDeploy, or customize your installation.

This document shows an inference sample on the CPU using the PaddleClas classification model MobileNetV2 as an example.

## 1. Obtaining the model

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz"
fd.download_and_decompress(model_url, path=".")
```

## 2. Backend Configuration

- For more examples, you can refer to [examples/runtime](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime).

``` python
option = fd.RuntimeOption()

option.set_model_path("mobilenetv2/inference.pdmodel",
                      "mobilenetv2/inference.pdiparams")

# **** CPU Configuration ****
option.use_cpu()
option.use_ort_backend()
option.set_cpu_thread_num(12)

# Initialise runtime
runtime = fd.Runtime(option)

# Get model input name
input_name = runtime.get_input_info(0).name

# Constructing random data for inference
results = runtime.infer({
    input_name: np.random.rand(1, 3, 224, 224).astype("float32")
})

print(results[0].shape)
```
When loading is complete, you will get the following output information indicating the initialized backend and the hardware devices.
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OrtBackend in device Device::CPU.
```

## Other Documents

- [Runtime demos on different backends](../../../../examples/runtime/README.md)
- [Switching hardware and backend for model inference](../../faq/how_to_change_backend.md)

