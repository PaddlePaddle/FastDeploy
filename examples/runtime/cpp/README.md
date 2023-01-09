English | [简体中文](README_CN.md)
# C++ Inference

Before running demo, the following two steps need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [Environment requirements for FastDeploy](../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Download pre-compiled libraries and samples according to the development environment. Please refer to [FastDeploy pre-compiled libraries](../../../docs/cn/build_and_install/download_prebuilt_libraries.md).

This document shows an inference example on the CPU using the PaddleClas classification model MobileNetV2 as an example.

## 1. Obtaining the Model

```bash
wget https://bj.bcebos.com/fastdeploy/models/mobilenetv2.tgz
tar xvf mobilenetv2.tgz
```

## 2. Backend Configuration

The following C++ code is saved as `infer_paddle_onnxruntime.cc`.

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
When loading is complete, the following prompt will be output, indicating the initialized backend, and the running hardware devices.
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OrtBackend in device Device::CPU.
```

## 3. Prepare for CMakeLists.txt

FastDeploy contains several dependencies, it is complicated to compile directly with `g++` or compiler, so we recommend using cmake for compiling configuration. The sample configuration is as follows:

```cmake
PROJECT(runtime_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

# Specify the path to the fastdeploy library after downloading and unpacking.
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# Add FastDeploy dependency headers.
include_directories(${FASTDEPLOY_INCS})

add_executable(runtime_demo ${PROJECT_SOURCE_DIR}/infer_onnx_openvino.cc)
# Adding FastDeploy library dependencies.
target_link_libraries(runtime_demo ${FASTDEPLOY_LIBS})
```

## 4. Compile executable program

Open the terminal, go to the directory where `infer_paddle_onnxruntime.cc` and `CMakeLists.txt` are located, and run the following command:

```bash
mkdir build & cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=$fastdeploy_cpp_sdk
make -j
```

```fastdeploy_cpp_sdk``` is path to FastDeploy C++ deployment libraries.

After compiling, run the following command and get the results.
```bash
./runtime_demo
```
If you are prompted with `error while loading shared libraries: libxxx.so: cannot open shared object file: No such file... `, it means that the path to FastDeploy libraries is not found, you can run the program again after adding the path to the environment variable by executing the following command.
```bash
source /Path/to/fastdeploy_cpp_sdk/fastdeploy_init.sh
```

This sample code is common on all platforms (Windows/Linux/Mac), but the compilation process is only supported on (Linux/Mac),while using msbuild to compile on Windows. Please refer to [FastDeploy C++ SDK on Windows](../../../docs/en/faq/use_sdk_on_windows.md).

## Other Documents

- [A Python example for Runtime](../python)
- [Switching hardware and backend for model inference](../../../docs/en/faq/how_to_change_backend.md)
