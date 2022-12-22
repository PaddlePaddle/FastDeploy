[English](../../../en/quick_start/models/cpp.md) | 中文

# C++部署

确认开发环境已准备FastDeploy C++部署库，参考[FastDeploy安装](../../build_and_install/)安装预编译的FastDeploy，或根据自己需求进行编译安装。

本文档以PaddleDetection目标检测模型PPYOLOE为例展示CPU上的推理示例

## 1. 获取模型和测试图像

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://bj.bcebos.com/fastdeploy/tests/test_det.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz
```

## 2. 准备C++推理代码

如下C++代码保存为`infer_demo.cc`

```c++
#include "fastdeploy/vision.h"
int main() {
  std::string model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  std::string params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  std::string infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  // 模型推理的配置信息
  fastdeploy::RuntimeOption option;
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file, infer_cfg_file, option);

  assert(model.Initialized()); // 判断模型是否初始化成功

  cv::Mat im = cv::imread("test_det.jpg");
  fastdeploy::vision::DetectionResult result;
  
  assert(model.Predict(&im, &result)); // 判断是否预测成功

  std::cout << result.Str() << std::endl;

  cv::Mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  // 可视化结果保存到本地
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result save in vis_result.jpg" << std::endl;
  return 0;
}
```

## 3. 准备CMakeList.txt

FastDeploy中包含多个依赖库，直接采用`g++`或编译器编译较为繁杂，推荐使用cmake进行编译配置。示例配置如下，

假设下载或准备的FastDeploy C++ SDK在`/Paddle/Download`目录下，且目录名为`fastdeploy_cpp_sdk`，在开发者的项目中只需添加如下代码，即可引入`FASTDEPLOY_INCS`和`FASTDEPLOY_LIBS`两个变量，分别表示依赖的头文件和库文件

```cmake
include(/Paddle/Download/fastdeploy_cpp_sdk/FastDeploy.cmake)
```

```cmake
PROJECT(infer_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.10)

include(/Path/to/fastdeploy_cpp_sdk/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(infer_demo ${PROJECT_SOURCE_DIR}/infer_demo.cc)
target_link_libraries(infer_demo ${FASTDEPLOY_LIBS})
```

## 4. 编译可执行程序

假设当前目录已经准备好`infer_demo.cc`和`CMakeLists.txt`两个文件，即可进行编译

### Linux & Mac

打开命令行终端，进入`infer_demo.cc`和`CmakeLists.txt`所在的目录，执行如下命令

```bash
mkdir build & cd build
cmake ..
make -j
```

编译完成后，使用如下命令执行可得到预测结果
```bash
./infer_demo 
```
执行时如提示`error while loading shared libraries: libxxx.so: cannot open shared object file: No such file...`，说明程序执行时没有找到FastDeploy的库路径，可通过执行如下命令，将FastDeploy的库路径添加到环境变量之后，重新执行二进制程序。
```bash
source /Path/to/fastdeploy_cpp_sdk/fastdeploy_init.sh
```

执行完屏幕会输出如下日志
```bash
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
415.047180,89.311569, 506.009613, 283.863098, 0.950423, 0
163.665710,81.914932, 198.585342, 166.760895, 0.896433, 0
581.788635,113.027618, 612.623474, 198.521713, 0.842596, 0
267.217224,89.777306, 298.796051, 169.361526, 0.837951, 0
...
...
```

同时可视化的检测结果图片保存在本地`vis_result.jpg`，查看效果如下
<div  align="center">
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

### Windows

上面的编译过程适用于Linux/Mac，Windows上编译流程如下

在Windows菜单中，找到`x64 Native Tools Command Prompt for VS 2019`打开，进入`infer_demo.cc`和`CMakeLists.txt`所在目录，执行如下命令
```bat
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
msbuild infer_demo.sln /m /p:Configuration=Release /p:Platform=x64
```

执行完后，即会在`build/Release`目录下生成`infer_demo.exe`程序，使用如下命令执行可得到预测结果
```bat
Release\infer_demo.exe
```

Windows上更多SDK使用方式参阅[Windows平台使用FastDeploy C++ SDK](../../faq/use_sdk_on_windows.md)

## 其它文档

- [切换模型推理的硬件和后端](../../faq/how_to_change_backend.md)
