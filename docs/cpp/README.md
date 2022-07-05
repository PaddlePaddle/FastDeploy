# C++部署

## 准备预测库

参考编译文档[FastDeploy编译](../compile/README.md)进行编译，或直接使用如下预编译库

| 编译库 | 平台 | 支持设备 | 说明 |
|:------ | :---- | :------- | :----- |
|[fastdeploy-linux-x64-0.0.3.tgz](https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz) | Linux | CPU | 集成ONNXRuntime |
|[fastdeploy-linux-x64-gpu-0.0.3.tgz](https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-gpu-0.0.3.tgz) | Linux | CPU/GPU | 集成ONNXRuntime, TensorRT |
|[fastdeploy-osx-x86_64-0.0.3.tgz](https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-osx-x86_64-0.0.3.tgz) | Mac OSX Intel CPU | CPU | 集成ONNXRuntime |
|[fastdeploy-osx-arm64-0.0.3.tgz](https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-osx-arm64-0.0.3.tgz) | Mac OSX M1 CPU | CPU | 集成ONNXRuntime |


## 使用

FastDeploy提供了多种领域内的模型，可快速完成模型的部署，本文档以YOLOv5在Linux上的部署为例

```
# 下载库并解压
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 下载模型和测试图片
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg
```

### YOLOv5预测代码

准备如下`yolov5.cc`代码
```
#include "fastdeploy/vision.h"

int main() {
  typedef vis = fastdeploy::vision;

  auto model = vis::ultralytics::YOLOv5("yolov5s.onnx"); // 加载模型

  if (!model.Initialized()) { // 判断模型是否初始化成功
    std::cerr << "Initialize failed." << std::endl;
    return -1;
  }

  cv::Mat im = cv::imread("bus.jpg"); // 读入图片

  vis::DetectionResult res;
  if (!model.Predict(&im, &res)) { // 预测图片
    std::cerr << "Prediction failed." << std::endl;
    return -1;
  }

  std::cout << res.Str() << std::endl; // 输出检测结果
  return 0;
}
```

### 编译代码

编译前先完成CMakeLists.txt的开发，在`yolov5.cc`同级目录创建`CMakeLists.txt`文件，内容如下
```
PROJECT(yolov5_demo C CXX)
CMAKE_MINIMUM_REQUIRED (VERSION 3.16)
# 在低版本ABI环境中，可通过如下代码进行兼容性编译
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 在下面指定下载解压后的fastdeploy库路径
set(FASTDEPLOY_INSTALL_DIR /ssd1/download/fastdeploy-linux-x64-0.0.3/)

include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

# 添加FastDeploy依赖头文件
include_directories(${FASTDEPLOY_INCS})

add_executable(yolov5_demo ${PROJECT_SOURCE_DIR}/yolov5.cc)
message(${FASTDEPLOY_LIBS})
# 添加FastDeploy库依赖
target_link_libraries(yolov5_demo ${FASTDEPLOY_LIBS})
~
```

此时当前目录结构如下所示
```
- demo_directory
|___fastdeploy-linux-x64-0.0.3/ # 预测库解压
|___yolov5.cc                   # 示例代码
|___CMakeLists.txt              # cmake文件
|___yolov5s.onnx                # 模型文件
|___bus.jpeg                    # 测试图片
```

执行如下命令进行编译
```
cmake .
make -j
```
编译后可执行二进制即为当前目录下的`yolov5_demo`，使用如下命令执行
```
./yolov5_demo
```

即会加载模型进行推理，得到结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
223.395126,403.948669, 345.337189, 867.339050, 0.856906, 0
668.301758,400.781372, 808.441772, 882.534973, 0.829716, 0
50.210758,398.571289, 243.123383, 905.016846, 0.805375, 0
23.768217,214.979355, 802.627869, 778.840820, 0.756311, 5
0.737200,552.281006, 78.617218, 890.945007, 0.363471, 0
```
