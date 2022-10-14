# C++ Deployment

Please make sure the development environment has FastDeploy C++ SDK installed. Refer to [FastDeploy installation](../../build_and_install/) to install the pre-built FastDeploy, or build and install according to your own needs.

This document shows an example to deploy a target detection model named PP-YOLOE, provided by PaddleDetection, with CPU.

## 1. Get the Model and Test Image

```
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz
```

## 2.Prepare C++ Inference Code

The following C++ code is saved as `infer_demo.cc`

``` c++
#include "fastdeploy/vision.h"
int main() {
   std::string model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  std::string params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  std::string infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  
  fastdeploy::RuntimeOption option; // Configuration information for model inference
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file, infer_cfg_file, option);

  assert(model.Initialized()); // Determine if the model is initialized successfully

  cv::mat im = cv::imread("000000014439.jpg");
  fastdeploy::vision::DetectionResult result;
  
  assert(model.Predict(im)); // Determine whether the prediction is successful

  std::cout << result << std::endl;

  cv::mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  
  cv::imwrite("vis_result.jpg", vis_im); // The visualization results are saved locally

  return 0;
}
```

## 3. Prepare CMakeList.txt


FastDeploy contains multiple dependent libraries. It is more complicated to compile directly with `g++` or the compiler. It is recommended to use cmake for compilation and configuration. An example configuration is as follows,

Assuming that the downloaded or prepared FastDeploy C++ SDK is in the `/Paddle/Download` directory, and the directory name is `fastdeploy_cpp_sdk`, you only need to add the following code to the developer's project to introduce `FASTDEPLOY_INCS` and `FASTDEPLOY_LIBS` Variables, representing dependent header files and library files, respectively

``` shell
include(/Paddle/Download/fastdeploy_cpp_sdk/FastDeploy.cmake)
```

```
PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

include(/Paddle/Download/fastdeploy_cpp_sdk/FastDeploy.cmake)

# Add FastDeploy dependency header files
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer_demo.cc)
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
```

## 4. Build the executable program


Assuming that the current directory has prepared two files `infer_demo.cc` and `CMakeLists.txt`, the directory structure is as follows, you can compile

### Linux & Mac

Open the command line terminal, enter the directory where `infer_demo.cc` and `CmakeLists.txt` are located, and execute the following command：

```
mkdir build & cd build
cmake ..
make -j
```

When executing the `cmake` command, the screen will output FastDeploy compilation information and Notice, among which the following prompts guide the developer to add the FastDeploy dependency library path to the environment variable, so that the binary program can be linked to the corresponding library after compilation, and the developer can copy The corresponding command can be executed in the terminal.


```
======================= Notice ========================
After compiled binary executable file, please add the following path to environment, execute the following command,

export LD_LIBRARY_PATH=/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/paddle2onnx/lib:/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/opencv/lib:/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/onnxruntime/lib:/Paddle/Download/fastdeploy_cpp_sdk/lib:${LD_LIBRARY_PATH}
=======================================================
```

After compiling, execute the following command to get the predicted result:

```
./infer_demo 
```

### Windows

Launch `the x64 Native Tools Command Prompt for VS 2019` from the Windows Start Menu and run the following commands:

```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
msbuild infer_demo.sln /m /p:Configuration=Release /p:Platform=x64
```


When executing the `cmake` command, the screen will output FastDeploy compilation information and Notice, among which the following prompts guide the developer to add the FastDeploy dependent library path to the environment variable, so that the exe can be linked to the corresponding library after compilation, and the developer can copy the corresponding library The command can be executed in the terminal.

```
======================= Notice ========================

```

After execution, the `infer_demo.exe` program will be generated in the `build/Release` directory, and the prediction result can be obtained by executing the following command:
```
Release\infer_demo.exe
```

For more SDK usage on Windows, please refer to [Using FastDeploy C++ SDK on Windows Platform](../../faq/use_sdk_on_windows.md)

## other documents

- [Switch model inference hardware and backend](../../faq/how_to_change_backend.md)
