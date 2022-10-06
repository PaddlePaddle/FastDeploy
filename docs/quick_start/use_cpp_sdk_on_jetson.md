# 在 Nvidia Jetson 上使用 FastDeploy C++ SDK

在 Jetson 上使用 FastDeploy C++ SDK ，目前暂时需要通过源码编译的方式获取到 C++ 的预测库。

下面以 PicoDet 在 Jetson Nano 上的部署为例进行演示，包括 CPU 推理 以及 GPU 上的 TensorRT 加速推理两部分。

同时也提供了基于 MIPI CSI 摄像头的 Demo。

## 目录

- [一、环境依赖](#Environment)   
- [二、Jetson 上编译 FastDeploy 的 C++ SDK](#Compiling)
- [三、准备模型文件和测试图片](#Download)
- [四、PicoDet 的 C++ 部署示例](#Deploy)
- [五、基于 MIPI CSI 摄像头的 Demo](#Camera)

<div id="Environment"></div>

## 一、环境依赖

Jetson 为 Linux aarch64 系统，采用 NVIDIA GPU；在其上使用 FastDeploy 安装好Jetpack 4.6.1后，以下环境就自动满足 ：

- jetpack = 4.6.1
- opencv = 4.1.1 compiled CUDA: No
- cmake >= 3.12
- gcc/g++ >= 8.2
- cuda >= 11.0 （Linux默认安装路径在/usr/local/cuda下）
- cudnn >= 8.0
- TensorRT、Paddle Inference、ONNXruntime等推理引擎，会在SDK中包含，不需要单独安装。

### 1. 安装必要的包

```
sudo apt-get install build-essential make cmake

sudo apt-get install git g++ pkg-config curl
```

### 2. 安装 jetson-stats 工具包

该工具包的 jtop 工具可以实时显示系统资源情况、检测 CPU 温度等。

```
sudo apt-get install python-pip

sudo -H pip install jetson-stats
```

<div id="Compiling"></div>

## 二、Jetson 上编译 FastDeploy 的 C++ SDK

### 1. 编译前，需安装 patchelf：

```
sudo apt-get install patchelf
```

### 2. 拉取 FastDeploy 的代码，并编译：

```
git clone https://github.com/PaddlePaddle/FastDeploy

cd FastDeploy

mkdir build && cd build

cmake .. -DBUILD_ON_JETSON=ON -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_cpp_sdk

make -j8

make install
```

编译产出的 C++ SDK 保存在 FastDeploy/build/fastdeploy_cpp_sdk 中，其中也包括 C++ 的实例代码。

<font size=3>**【提示】**</font>

FastDeploy 在Jetson上编译会依赖第三方库 Eigen，在编译过程中如遇自动拉取 GitHub 上Eigen 源码失败问题，可先使用如下命令配置 git：

```
git config --global http.sslverify false
```

<div id="Download"></div>

## 三、准备模型文件和测试图片

- 基于 COCO 数据集训练并导出后的 PicoDet-s-416 模型可[点此下载](https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar)；使用FastDeploy SDK时，导出模型注意事项参考examples下的模型说明，例如 PaddleDetection 中的模型到处参考[对应说明](../../examples/vision/detection/paddledetection#导出部署模型)。

- 测试图片可[点此下载](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/demo/000000014439.jpg)

<div id="Deploy"></div>

## 四、PicoDet 的 C++ 部署示例

### 1. 把依赖库导入环境变量

FastDeploy 编译成功后会生成把依赖库导入环境变量的脚本：fastdeploy_init.sh，

脚本位于：

```
YourPathTo/fastdeploy_cpp_sdk/
```

在此目录下，直接执行此命令，即可完成导入：

```
source fastdeploy_init.sh
```

### 2. 编译 C++ Demo

* FastDeploy 提供的示例代码位于：

```
YourPathTo/fastdeploy_cpp_sdk/examples/vision/detection/paddledetection/cpp/infer_picodet.cc
```

* 进入以上的目录后，依次运行如下编译命令：

```
mkdir build && cd build

cmake .. -DFASTDEPLOY_INSTALL_DIR=YourPathTo/fastdeploy_cpp_sdk

make -j
```

* 进入到 YourPathTo/fastdeploy_cpp_sdk/examples/vision/detection/paddledetection/cpp/build 目录，可找到编译后的可执行文件： infer_picodet_demo

* 将导出后的 PicoDet 模型和测试图片拷贝到当前build目录下

### 3. CPU 推理

```
./infer_picodet_demo ./PicoDet-s-416-DHQ-exported trash01.png 0
```

推理结果的图片，会保存在可执行文件的同级目录下。

PicoDet目标检测模型，推理框上的text内容为 {类别ID，置信度}

### 4. GPU+TensorRT 推理

```
./infer_picodet_demo ./PicoDet-s-416-DHQ-exported trash01.png 2
```

FastDeploy 默认采用TRT-FP32的推理。如果需要使用TRT-FP16的推理，只需要在代码中加入一行 option.EnableTrtFP16() 即可。

![](https://ai-studio-static-online.cdn.bcebos.com/25ca48f7ecd643ac99715c1db68183bc5a245b51a355412f803f391d87c4ed31)

<font size=3>**【注意】**</font>

编译 FastDeploy 时，当打开开关 BUILD_ON_JETSON 时，会默认开启 ENABLE_ORT_BACKEND 和 ENABLE_TRT_BACKEND，即当前仅支持 ONNXRuntime CPU 或 TensorRT 两种后端分别用于在 CPU 和 GPU 上的推理。因此，这里不带 TensorRT 的 GPU 推理并不会生效，而是会自动转成 CPU 推理。

<font size=3>**【提示】**</font>

为避免每次开启 TensorRT 推理时，初始化时间过长，可以在代码中加入一行：

```
option.SetTrtCacheFile("./picodet.trt")
```

这样，第一次初始化完成之后，会在当前目录下保存 TensorRT 的缓存文件为 picodet.trt，这样以后每次运行就直接读取该文件了，避免重复初始化。

<font size=3>**【注意】**</font>

当需要在 TensorRT-FP32 和 TensorRT-FP16 之间切换时，要先删除之前保存的 picodet.trt 缓存文件。

<div id="Camera"></div>

## 五、基于 MIPI CSI 摄像头的 Demo

首先，使用如下命令测试 Jetson 开发板上摄像头是否可以正常开启：

```
gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e
```

CSI 接口的摄像头 使用 C++ 调用的简单示意如下：

```
// 定义gstreamer pipeline
std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{ // DHQ added 20220927
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


。。。


// 在主函数中设置参数，并调用gstreamer管道
int capture_width = 1280 ;
int capture_height = 720 ;
int display_width = 1280 ;
int display_height = 720 ;
int framerate = 60 ;
int flip_method = 0 ;

//创建管道
std::string pipeline = gstreamer_pipeline(capture_width,
capture_height,
display_width,
display_height,
framerate,
flip_method);
std::cout << "使用gstreamer管道: \n\t" << pipeline << "\n";

//管道与视频流绑定
cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);


// 然后再 cap.read() 就可以了。
```

完整的代码请参考aistudio项目中 `~/work/infer_picodet_camera.cc` 代码。

<font size=3>**【感谢】**</font>

感谢 @Taichipeace 贡献该文档，并进行全流程的完整验证。
