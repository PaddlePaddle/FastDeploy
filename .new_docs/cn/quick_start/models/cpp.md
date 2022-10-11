# C++部署

确认开发环境已准备FastDeploy C++部署库，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以PaddleDetection目标检测模型PPYOLOE为例展示CPU上的推理示例

## 1. 获取模型和测试图像

```
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz
```

## 2. 准备C++推理代码

如下C++代码保存为`infer_demo.cc`

``` c++
#include "fastdeploy/vision.h"
int main() {
  std::string model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  std::string params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  std::string infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file, infer_cfg_file);

  assert(model.Initialized()); // 判断模型是否初始化成功

  cv::mat im = cv::imread("000000014439.jpg");
  fastdeploy::vision::DetectionResult result;
  
  assert(model.Predict(im)); // 判断是否预测成功

  std::cout << result << std::endl;

  cv::mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  // 可视化结果保存到本地
  cv::imwrite("vis_result.jpg", vis_im);

  return 0;
}
```

## 3. 准备CMakeList.txt

FastDeploy中包含多个依赖库，直接采用`g++`或编译器编译较为繁杂，推荐使用cmake进行编译配置。示例配置如下，

假设下载或准备的FastDeploy C++ SDK在`/Paddle/Download`目录下，且目录名为`fastdeploy_cpp_sdk`，在开发者的项目中只需添加如下代码，即可引入`FASTDEPLOY_INCS`和`FASTDEPLOY_LIBS`两个变量，分别表示依赖的头文件和库文件

``` shell
include(/Paddle/Download/fastdeploy_cpp_sdk/FastDeploy.cmake)
```

```
PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

include(/Paddle/Download/fastdeploy_cpp_sdk/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer_demo.cc)
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
```

## 4. 编译可执行程序

假设当前目录已经准备好`infer_demo.cc`和`CMakeLists.txt`两个文件，目录结构如下所示，即可进行编译

### Linux & Mac

打开命令行终端，进入`infer_demo.cc`和`CmakeLists.txt`所在的目录，执行如下命令

```
mkdir build & cd build
cmake ..
make -j
```

在执行`cmake`命令时，屏幕会输出FastDeploy编译信息以及Notice，其中如下提示是指引开发者将FastDeploy依赖库路径添加到环境变量，便于编译后执行二进制程序能链接到相应的库，开发者复制相应command在终端执行即可。

```
======================= Notice ========================
After compiled binary executable file, please add the following path to environment, execute the following command,

export LD_LIBRARY_PATH=/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/paddle2onnx/lib:/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/opencv/lib:/Paddle/Download/fastdeploy_cpp_sdk/third_libs/install/onnxruntime/lib:/Paddle/Download/fastdeploy_cpp_sdk/lib:${LD_LIBRARY_PATH}
=======================================================
```

编译完成后，使用如下命令执行可得到预测结果
```
./infer_demo 
```

### Windows

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，进入`infer_demo.cc`和`CMakeLists.txt`所在目录，执行如下命令
```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
msbuild infer_demo.sln /m /p:Configuration=Release /p:Platform=x64
```

在执行`cmake`命令时，屏幕会输出FastDeploy编译信息以及Notice，其中如下提示是指引开发者将FastDeploy依赖库路径添加到环境变量，便于编译后执行exe能链接到相应的库，开发者复制相应command在终端执行即可。

```
======================= Notice ========================

```

执行完后，即会在`build/Release`目录下生成`infer_demo.exe`程序，使用如下命令执行可得到预测结果
```
Release\infer_demo.exe
```
