简体中文 ｜ [English](README.md)
# Python推理

在运行demo前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本文档以 PaddleClas 分类模型 MobileNetV2 为例展示 CPU 上的推理示例

## 1. 获取模型

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz"
fd.download_and_decompress(model_url, path=".")
```

## 2. 配置后端

``` python
option = fd.RuntimeOption()

option.set_model_path("mobilenetv2/inference.pdmodel",
                      "mobilenetv2/inference.pdiparams")

# **** CPU 配置 ****
option.use_cpu()
option.use_ort_backend()
option.set_cpu_thread_num(12)

# 初始化构造runtime
runtime = fd.Runtime(option)

# 获取模型输入名
input_name = runtime.get_input_info(0).name

# 构造随机数据进行推理
results = runtime.infer({
    input_name: np.random.rand(1, 3, 224, 224).astype("float32")
})

print(results[0].shape)
```
加载完成，会输出提示如下，说明初始化的后端，以及运行的硬件设备
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OrtBackend in device Device::CPU.
```

## 其它文档

- [Runtime C++ 示例](../cpp)
- [切换模型推理的硬件和后端](../../../docs/cn/faq/how_to_change_backend.md)
