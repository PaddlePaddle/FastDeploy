English | [中文](../../../cn/quick_start/runtime/cpp.md)
# C++ Inference

Please check out the FastDeploy C++ deployment library is already in your environment. You can refer to [FastDeploy Installation](../../build_and_install/) to install the pre-compiled FastDeploy, or customize your installation.

This document shows an inference sample on the CPU using the PaddleClas classification model MobileNetV2 as an example.

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
When loading is complete, you can get the following output information indicating the initialized backend and the hardware devices.
```
[INFO] fastdeploy/fastdeploy_runtime.cc(283)::Init	Runtime initialized with Backend::OrtBackend in device Device::CPU.
```

## 3. Prepare for CMakeLists.txt

FastDeploy contains several dependencies, it is more complicated to compile directly with `g++` or a compiler, so we recommend to use cmake to compile and configure. The sample configuration is as follows.

```cmake
PROJECT(runtime_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

# Specify path to the fastdeploy library after downloading and unpacking
option(FASTDEPLOY_INSTALL_DIR "Path of downloaded fastdeploy sdk.")

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# Add FastDeploy dependency headers
include_directories(${FASTDEPLOY_INCS})

add_executable(runtime_demo ${PROJECT_SOURCE_DIR}/infer_onnx_openvino.cc)
# Add FastDeploy dependency libraries
target_link_libraries(runtime_demo ${FASTDEPLOY_LIBS})
```

## 4. Compile executable program

Open a terminal, go to the directory where `infer_paddle_onnxruntime.cc` and `CMakeLists.txt` are located, and then run:

```bash
cd examples/runtime/cpp
mkdir build & cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=$fastdeploy_cpp_sdk
make -j
```

```fastdeploy_cpp_sdk``` is path to FastDeploy C++ deployment library.

After compiling, you can get your results by running:
```bash
./runtime_demo
```
If `error while loading shared libraries: libxxx.so: cannot open shared object file: No such file...`is reported, it means that the path to FastDeploy is not found. You can re-execute the program after adding the  FastDeploy library path to the environment variable by running the following command.
```bash
source /Path/to/fastdeploy_cpp_sdk/fastdeploy_init.sh
```

This sample code is common on all platforms (Windows/Linux/Mac), but the compilation process is only supported on (Linux/Mac),while using msbuild to compile on Windows. Please refer to [FastDeploy C++ SDK on Windows](../../faq/use_sdk_on_windows.md).

## Other Documents

- [Runtime demos on different backends](../../../../examples/runtime/README.md)
- [Switching hardware and backend for model inference](../../faq/how_to_change_backend.md)
