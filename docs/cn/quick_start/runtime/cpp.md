# C++推理

确认开发环境已准备FastDeploy C++部署库，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以 PaddleClas 分类模型 MobileNetV2 为例展示CPU上的推理示例

## 1. 获取模型

```bash
wget https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz
tar xvf mobilenetv2.tgz
```

## 2. 配置后端

如下C++代码保存为`infer_paddle_onnxruntime.cc`

``` c++
#include "fastdeploy/runtime.h"

namespace fd = fastdeploy;

int main(int argc, char* argv[]) {
  std::string model_file = "mobilenetv2/inference.pdmodel";
  std::string params_file = "mobilenetv2/inference.pdiparams";

  // setup option
  fd::RuntimeOption runtime_option;
  runtime_option.SetModelPath(model_file, params_file, fd::ModelFormat::PADDLE);
  runtime_option.UseOrtBackend();
  runtime_option.SetCpuThreadNum(12);
  // init runtime
  std::unique_ptr<fd::Runtime> runtime =
      std::unique_ptr<fd::Runtime>(new fd::Runtime());
  if (!runtime->Init(runtime_option)) {
    std::cerr << "--- Init FastDeploy Runitme Failed! "
              << "\n--- Model:  " << model_file << std::endl;
    return -1;
  } else {
    std::cout << "--- Init FastDeploy Runitme Done! "
              << "\n--- Model:  " << model_file << std::endl;
  }
  // init input tensor shape
  fd::TensorInfo info = runtime->GetInputInfo(0);
  info.shape = {1, 3, 224, 224};

  std::vector<fd::FDTensor> input_tensors(1);
  std::vector<fd::FDTensor> output_tensors(1);

  std::vector<float> inputs_data;
  inputs_data.resize(1 * 3 * 224 * 224);
  for (size_t i = 0; i < inputs_data.size(); ++i) {
    inputs_data[i] = std::rand() % 1000 / 1000.0f;
  }
  input_tensors[0].SetExternalData({1, 3, 224, 224}, fd::FDDataType::FP32, inputs_data.data());

  //get input name
  input_tensors[0].name = info.name;

  runtime->Infer(input_tensors, &output_tensors);

  output_tensors[0].PrintInfo();
  return 0;
}
```
加载完成，会输出提示如下，说明初始化的后端，以及运行的硬件设备
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OrtBackend in device Device::CPU.
```

## 3. 准备CMakeLists.txt

FastDeploy中包含多个依赖库，直接采用`g++`或编译器编译较为繁杂，推荐使用cmake进行编译配置。示例配置如下，

```cmake
PROJECT(runtime_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

# 指定下载解压后的fastdeploy库路径
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(runtime_demo ${PROJECT_SOURCE_DIR}/infer_onnx_openvino.cc)
# 添加FastDeploy库依赖
target_link_libraries(runtime_demo ${FASTDEPLOY_LIBS})
```

## 4. 编译可执行程序

打开命令行终端，进入`infer_paddle_onnxruntime.cc`和`CMakeLists.txt`所在的目录，执行如下命令

```bash
cd examples/runtime/cpp
mkdir build & cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=$fastdeploy_cpp_sdk
make -j
```

```fastdeploy_cpp_sdk``` 为FastDeploy C++部署库路径

编译完成后，使用如下命令执行可得到预测结果
```bash
./runtime_demo
```
执行时如提示`error while loading shared libraries: libxxx.so: cannot open shared object file: No such file...`，说明程序执行时没有找到FastDeploy的库路径，可通过执行如下命令，将FastDeploy的库路径添加到环境变量之后，重新执行二进制程序。
```bash
source /Path/to/fastdeploy_cpp_sdk/fastdeploy_init.sh
```

本示例代码在各平台(Windows/Linux/Mac)上通用，但编译过程仅支持(Linux/Mac)，Windows上使用msbuild进行编译，具体使用方式参考[Windows平台使用FastDeploy C++ SDK](../../faq/use_sdk_on_windows.md)

## 其它文档

- [不同后端Runtime demo示例](../../../../examples/runtime/README.md)
- [切换模型推理的硬件和后端](../../faq/how_to_change_backend.md)
