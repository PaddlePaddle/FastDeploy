# 简介

本文档以[千分类模型_MobileNetV3](https://ai.baidu.com/easyedge/app/openSource)为例，介绍FastDeploy中的模型SDK ，在**Intel x86_64 / NVIDIA GPU Windows C++** 环境下：（1）SDK 图像和视频推理部署步骤；（2）介绍模型推流全流程API，方便开发者了解项目后二次开发。
其中Windows Python请参考[Windows Python环境下的推理部署](./Windows-Python-SDK-Inference.md)文档。

<!--ts-->

* [简介](#简介)

* [环境准备](#环境准备)
  
  * [1. SDK下载](#1-sdk下载)
  * [2. CPP环境](#2-cpp环境)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. 测试EasyEdge服务](#2-测试easyedge服务)
  * [3. 预测图像](#3-预测图像)
  * [4. 预测视频流](#4-预测视频流)
  * [5. 编译Demo](#5-编译demo)

* [预测API流程详解](#预测api流程详解)
  
  * [1. SDK参数运行配置](#1-sdk参数运行配置)
  * [2. 初始化Predictor](#2-初始化predictor)
  * [3. 预测推理](#3-预测推理)
    * [3.1 预测图像](#31-预测图像)
    * [3.2 预测视频](#32-预测视频)

* [FAQ](#faq)
  
  <!--te-->

# 环境准备

## 1. SDK下载

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压缩后的文件结构如`快速开始`中[1项目介绍说明](#1-项目结构说明)介绍。

## 2. CPP环境

> 建议使用Microsoft Visual Studio 2015及以上版本，获取核心 C 和 C++ 支持，安装时请选择“使用 C++ 的桌面开发”工作负载。

# 快速开始

## 1. 项目结构说明

```shell
EasyEdge-win-xxx
        ├── data
        │   ├── model    # 模型文件资源文件夹，可替换为其他模型
        │   └── config   # 配置文件
        ├── bin          # demo二进制程序
           │   ├── xxx_image    # 预测图像demo
           │   ├── xxx_video    # 预测视频流demo
        │  └── xxx_serving  # 启动http预测服务demo
        ├── dll          # demo二进制程序依赖的动态库
        ├── ...          # 二次开发依赖的文件
        ├── python       # Python SDK文件
        ├── EasyEdge.exe # EasyEdge服务
        └── README.md    # 环境说明
```

## 2. 测试EasyEdge服务

> 模型资源文件默认已经打包在开发者下载的SDK包中，请先将zip包整体拷贝到具体运行的设备中，再解压缩使用。

SDK下载完成后，双击打开EasyEdge.exe启动推理服务，输入要绑定的Host ip及端口号Port，点击启动服务。

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854086-d507c288-56c8-4fa9-a00c-9d3cfeaac1c8.png" alt="图片" style="zoom: 67%;" />
</div>

服务启动后，打开浏览器输入`http://{Host ip}:{Port}`，添加图片或者视频来进行测试。

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854073-fb8189e5-0ffb-472c-a17d-0f35aa6a8418.png" style="zoom:67%;" />
</div>

## 3. 预测图像

除了通过上述方式外，您还可以使用bin目录下的可执行文件来体验单一的功能。在dll目录下，点击右键，选择"在终端打开"，执行如下命令。

> 需要将bin目录下的可执行文件移动到dll目录下执行，或者将dll目录添加到系统环境变量中。

```bash
.\easyedge_image_inference {模型model文件夹}  {测试图片路径}
```

运行效果示例：

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175854068-28d27c0a-ef83-43ee-9e89-b65eed99b476.jpg" width="400"></div>

```shell
2022-06-20 10:36:57,602 INFO [EasyEdge] 9788 EasyEdge Windows Development Kit 1.5.2(Build CPU.Generic 20220607) Release
e[37m---    Fused 0 subgraphs into layer_norm op.e[0m
2022-06-20 10:36:58,008 INFO [EasyEdge] 9788 Allocate graph success.
Results of image ..\demo.jpg:
8, n01514859 hen, p:0.953429
save result image to ..\demo.jpg.result-cpp.jpg
Done
```

可以看到，运行结果为`index：8，label：hen`，通过imagenet [类别映射表](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)，可以找到对应的类别，即 'hen'，由此说明我们的预测结果正确。

## 4. 预测视频流

```
.\easyedge_video_inference {模型model文件夹} {video_type} {video_src}
```

其中video_type支持三种视频流类型，它们分别是：（1）本地视频文件 （2）本地摄像头id（3）网络视频流地址。

```
/**
 * @brief 输入源类型
 */
enum class SourceType {
    kVideoFile = 1,                   // 本地视频文件
    kCameraId = 2,                    // 摄像头的index
    kNetworkStream = 3,               // 网络视频流
};
```

video_src 即为文件路径。

## 5. 编译Demo

在[项目结构说明](#1-项目结构说明)中，`bin`路径下的可执行文件是由`src`下的对应文件编译得到的，具体的编译命令如下。

```
cd src
mkdir build && cd build
cmake .. && make
```

编译完成后，在build文件夹下会生成编译好的可执行文件，如图像推理的二进制文件：`build/demo_serving/easyedge_serving`。

# 预测API流程详解

本章节主要结合前文的Demo示例来介绍推理API，方便开发者学习并将运行库嵌入到开发者的程序当中，更详细的API请参考`include/easyedge/easyedge*.h`文件。图像、视频的推理包含以下3个API，查看下面的cpp代码中的step注释说明。

> ❗注意：  
> （1）`src`文件夹中包含完整可编译的cmake工程实例，建议开发者先行了解[cmake工程基本知识](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)。  
> （2）请优先参考SDK中自带的Demo工程的使用流程和说明。遇到错误，请优先参考文件中的注释、解释、日志说明。

```cpp
    // step 1: SDK配置运行参数
    EdgePredictorConfig config;
    config.model_dir = {模型文件目录};

    // step 2: 创建并初始化Predictor；这这里选择合适的引擎
    auto predictor = global_controller()->CreateEdgePredictor(config);

    // step 3-1: 预测图像
    auto img = cv::imread({图片路径});
    std::vector<EdgeResultData> results;
    predictor->infer(img, results);

    // step 3-2: 预测视频
    std::vector<EdgeResultData> results;
    FrameTensor frame_tensor;
    VideoConfig video_config;
    video_config.source_type = static_cast<SourceType>(video_type);  // source_type 定义参考头文件 easyedge_video.h
    video_config.source_value = video_src;
    /*
    ... more video_configs, 根据需要配置video_config的各选项
    */
    auto video_decoding = CreateVideoDecoding(video_config);
    while (video_decoding->next(frame_tensor) == EDGE_OK) {
        results.clear();
        if (frame_tensor.is_needed) {
            predictor->infer(frame_tensor.frame, results);
            render(frame_tensor.frame, results, predictor->model_info().kind);
        }
        //video_decoding->display(frame_tensor); // 显示当前frame，需在video_config中开启配置
        //video_decoding->save(frame_tensor); // 存储当前frame到视频，需在video_config中开启配置
     }
```

若需自定义library search path或者gcc路径，修改对应Demo工程下的CMakeList.txt即可。

## 1. SDK参数运行配置

SDK的参数通过`EdgePredictorConfig::set_config`和`global_controller()->set_config`配置。本Demo 中设置了模型路径，其他参数保留默认参数。更详细的支持运行参数等，可以参考开发工具包中的头文件（`include/easyedge/easyedge_xxxx_config.h`）的详细说明。

配置参数使用方法如下：

```
EdgePredictorConfig config;
config.model_dir = {模型文件目录};
```

## 2. 初始化Predictor

- 接口
  
  ```cpp
  auto predictor = global_controller()->CreateEdgePredictor(config);
  predictor->init();
  ```

若返回非0，请查看输出日志排查错误原因。

## 3. 预测推理

### 3.1 预测图像

> 在Demo中展示了预测接口infer()传入cv::Mat& image图像内容，并将推理结果赋值给std::vector& result。更多关于infer()的使用，可以根据参考`easyedge.h`头文件中的实际情况、参数说明自行传入需要的内容做推理

- 接口输入

```cpp
 /**
  * @brief
  * 通用接口
  * @param image: must be BGR , HWC format (opencv default)
  * @param result
  * @return
  */
 virtual int infer(cv::Mat& image, std::vector<EdgeResultData>& result) = 0;
```

图片的格式务必为opencv默认的BGR, HWC格式。

- 接口返回
  
  `EdgeResultData`中可以获取对应的分类信息、位置信息。

```cpp
struct EdgeResultData {
    int index;  // 分类结果的index
    std::string label;  // 分类结果的label
    float prob;  // 置信度

    // 物体检测 或 图像分割时使用：
    float x1, y1, x2, y2;  // (x1, y1): 左上角， （x2, y2): 右下角； 均为0~1的长宽比例值。

    // 图像分割时使用：
    cv::Mat mask;  // 0, 1 的mask
    std::string mask_rle;  // Run Length Encoding，游程编码的mask
};
```

*** 关于矩形坐标 ***

x1 * 图片宽度 = 检测框的左上角的横坐标

y1 * 图片高度 = 检测框的左上角的纵坐标

x2 * 图片宽度 = 检测框的右下角的横坐标

y2 * 图片高度 = 检测框的右下角的纵坐标

*** 关于图像分割mask ***

```
cv::Mat mask为图像掩码的二维数组
{
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
}
其中1代表为目标区域，0代表非目标区域
```

*** 关于图像分割mask_rle ***

该字段返回了mask的游程编码，解析方式可参考 [http demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)。

以上字段可以参考demo文件中使用opencv绘制的逻辑进行解析。

### 3.2 预测视频

SDK 提供了支持摄像头读取、视频文件和网络视频流的解析工具类`VideoDecoding`，此类提供了获取视频帧数据的便利函数。通过`VideoConfig`结构体可以控制视频/摄像头的解析策略、抽帧策略、分辨率调整、结果视频存储等功能。对于抽取到的视频帧可以直接作为SDK infer 接口的参数进行预测。

- 接口输入

class`VideoDecoding`：

```
    /**
     * @brief 获取输入源的下一帧
     * @param frame_tensor
     * @return
     */
    virtual int next(FrameTensor &frame_tensor) = 0;

    /**
     * @brief 显示当前frame_tensor中的视频帧
     * @param frame_tensor
     * @return
     */
    virtual int display(const FrameTensor &frame_tensor) = 0;

    /**
     * @brief 将当前frame_tensor中的视频帧写为本地视频文件
     * @param frame_tensor
     * @return
     */
    virtual int save(FrameTensor &frame_tensor) = 0;

    /**
     * @brief 获取视频的fps属性
     * @return
     */
    virtual int get_fps() = 0;
     /**
      * @brief 获取视频的width属性
      * @return
      */
    virtual int get_width() = 0;

    /**
     * @brief 获取视频的height属性
     * @return
     */
    virtual int get_height() = 0;
```

struct `VideoConfig`

```
/**
 * @brief 视频源、抽帧策略、存储策略的设置选项
 */
struct VideoConfig {
    SourceType source_type;            // 输入源类型
    std::string source_value;          // 输入源地址，如视频文件路径、摄像头index、网络流地址
    int skip_frames{0};                // 设置跳帧，每隔skip_frames帧抽取一帧，并把该抽取帧的is_needed置为true
    int retrieve_all{false};           // 是否抽取所有frame以便于作为显示和存储，对于不满足skip_frames策略的frame，把所抽取帧的is_needed置为false
    int input_fps{0};                  // 在采取抽帧之前设置视频的fps
    Resolution resolution{Resolution::kAuto}; // 采样分辨率，只对camera有效

    bool enable_display{false};         // 默认不支持。
    std::string window_name{"EasyEdge"};
    bool display_all{false};           // 是否显示所有frame，若为false，仅显示根据skip_frames抽取的frame

    bool enable_save{false};
    std::string save_path;             // frame存储为视频文件的路径
    bool save_all{false};              // 是否存储所有frame，若为false，仅存储根据skip_frames抽取的frame

    std::map<std::string, std::string> conf;
};
```

| 序号  | 字段             | 含义                                                                                                                                 |
| --- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `source_type`  | 输入源类型，支持视频文件、摄像头、网络视频流三种，值分别为1、2、3                                                                                                 |
| 2   | `source_value` | 若`source_type`为视频文件，该值为指向视频文件的完整路径；若`source_type`为摄像头，该值为摄像头的index，如对于`/dev/video0`的摄像头，则index为0；若`source_type`为网络视频流，则为该视频流的完整地址。 |
| 3   | `skip_frames`  | 设置跳帧，每隔skip_frames帧抽取一帧，并把该抽取帧的is_needed置为true，标记为is_needed的帧是用来做预测的帧。反之，直接跳过该帧，不经过预测。                                             |
| 4   | `retrieve_all` | 若置该项为true，则无论是否设置跳帧，所有的帧都会被抽取返回，以作为显示或存储用。                                                                                         |
| 5   | `input_fps`    | 用于抽帧前设置fps                                                                                                                         |
| 6   | `resolution`   | 设置摄像头采样的分辨率，其值请参考`easyedge_video.h`中的定义，注意该分辨率调整仅对输入源为摄像头时有效                                                                       |
| 7   | `conf`         | 高级选项。部分配置会通过该map来设置                                                                                                                |

*** 注意：***

1. `VideoConfig`不支持`display`功能。如果需要使用`VideoConfig`的`display`功能，需要自行编译带有GTK选项的OpenCV。

2. 使用摄像头抽帧时，如果通过`resolution`设置了分辨率调整，但是不起作用，请添加如下选项：
   
   ```
   video_config.conf["backend"] = "2";
   ```

3. 部分设备上的CSI摄像头尚未兼容，如遇到问题，可以通过工单、QQ交流群或微信交流群反馈。

具体接口调用流程，可以参考SDK中的`demo_video_inference`。

# FAQ

1. 执行infer_demo文件时，提示your generated code is out of date and must be regenerated with protoc >= 3.19.0

进入当前项目，首先卸载protobuf

```shell
python3 -m pip uninstall protobuf
```

安装低版本protobuf

```shell
python3 -m pip install protobuf==3.19.0
```
