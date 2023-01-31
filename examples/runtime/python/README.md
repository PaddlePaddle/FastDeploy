English | [简体中文](README_CN.md)
# Python Inference

Before running demo, the following two steps need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [Environment requirements for FastDeploy](../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. Install FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../docs/cn/build_and_install/download_prebuilt_libraries.md).

This document shows an inference example on the CPU using the PaddleClas classification model MobileNetV2 as an example.

## 1. Obtaining the model

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz"
fd.download_and_decompress(model_url, path=".")
```

## 2. Backend Configuration

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

- [A C++ example for Runtime C++](../cpp)
- [Switching hardware and backend for model inference](../../../docs/en/faq/how_to_change_backend.md)
