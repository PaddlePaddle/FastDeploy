# Python推理

确认开发环境已安装FastDeploy，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以 PaddleClas 分类模型 MobileNetV2 为例展示CPU上的推理示例

## 1. 获取模型

``` python
import fastdeploy as fd

model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz"
fd.download_and_decompress(model_url, path=".")
```

## 2. 配置后端

- 更多后端的示例可参考[examples/runtime](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime)

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

- [不同后端Runtime demo示例](../../../../examples/runtime/README.md)
- [切换模型推理的硬件和后端](../../faq/how_to_change_backend.md)
