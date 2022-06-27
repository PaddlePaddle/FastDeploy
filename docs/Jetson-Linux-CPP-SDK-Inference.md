# 简介

本文档介绍FastDeploy中的模型SDK， 在**Jetson Linux C++** 环境下：（1） 图像和视频 推理部署步骤， （2）介绍推理全流程API，方便开发者了解项目后二次开发。如果开发者对Jetson的服务化部署感兴趣，可以参考[Jetson CPP Serving](./Jetson-Linux-CPP-SDK-Serving.md)文档。

**注意**：OCR目前只支持**图像**推理部署。

<!--ts-->

* [简介](#简介)

* [环境要求](#环境要求)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. 测试Demo](#2-测试demo)
    * [2.1 预测图像](#21-预测图像)
    * [2.2 预测视频流](#22-预测视频流)

* [预测API流程详解](#预测api流程详解)
  
  * [1. SDK参数运行配置](#1-sdk参数运行配置)
  * [2. 初始化Predictor](#2-初始化predictor)
  * [3. 预测推理](#3-预测推理)
    * [3.1 预测图像](#31-预测图像)
    * [3.2 预测视频](#32-预测视频)

* [FAQ](#faq)
  
  <!--te-->

# 环境要求

* Jetpack: 4.6，安装Jetpack，参考[NVIDIA 官网-Jetpack4.6安装指南](https://developer.nvidia.com/jetpack-sdk-46)，或者参考采购的硬件厂商提供的安装方式进行安装。![]()
  
  | 序号  | 硬件               | Jetpack安装方式        | 下载链接                                                                                                                                        | ---- |
  | --- | ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
  | 1   | Jetson Xavier NX | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jetson_xavier_nx/jetson-nx-jp46-sd-card-image.zip)      | ---- |
  | 2   | Jetson Nano      | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano/jetson-nano-jp46-sd-card-image.zip)         | ---- |
  | 3   | Jetson Nano 2GB  | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano_2gb/jetson-nano-2gb-jp46-sd-card-image.zip) | ---- |
  | 4   | agx xavier等      | NVIDIA SDK Manager | [Download NVIDIA SDK](https://developer.nvidia.com/nvsdk-manager)                                                                           | ---- |
  | 5   | 非官方版本，如emmc版     | 参考采购的硬件公司提供的安装指南   | ----                                                                                                                                        | ---- |
  
  注意：本项目SDK要求 `CUDA=10.2`、`cuDNN=8.2`、`TensorRT=8.0`、`gcc>=7.5` 、`cmake 在 3.0以上` ，安装 Jetpack4.6系统包后，CUDA、cuDNN、TensorRT、gcc和cmake版本就已经满足要求，无需在进行安装。

# 快速开始

## 1. 项目结构说明

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压后SDK目录结构如下：

```
.EasyEdge-Linux-硬件芯片
├── RES  # 模型资源文件夹，一套模型适配不同硬件、OS和部署方式
│   ├── conf.json        # Android、iOS系统APP名字需要
│   ├── model            # 模型结构文件 
│   ├── params           # 模型参数文件
│   ├── label_list.txt   # 模型标签文件
│   ├── infer_cfg.json   # 模型前后处理等配置文件
├── ReadMe.txt
├── cpp                  # C++ SDK 文件结构
    └── baidu_easyedge_linux_cpp_x86_64_CPU.Generic_gcc5.4_v1.4.0_20220325.tar.gz
        ├── ReadMe.txt   
        ├── bin          # 可直接运行的二进制文件
        ├── include      # 二次开发用的头文件 
        ├── lib          # 二次开发用的所依赖的库
        ├── src          # 二次开发用的示例工程
        └── thirdparty   # 第三方依赖
```

## 2. 测试Demo

> 模型资源文件（即压缩包中的RES文件夹）默认已经打包在开发者下载的SDK包中，请先将tar包整体拷贝到具体运行的设备中，再解压缩使用。

SDK中已经包含预先编译的二进制，可直接运行。以下运行示例均是`cd cpp/bin`路径下执行的结果。

### 2.1 预测图像

```bash
./easyedge_image_inference {模型RES文件夹路径}  {测试图片路径}
```

运行效果示例：

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175855351-68d1a4f0-6226-4484-b190-65f1ac2c7128.png" width="400"></div>

```bash
 > ./easyedge_image_inference ../../../../RES 2.jpeg
2019-02-13 16:46:12,659 INFO [EasyEdge] [easyedge.cpp:34] 140606189016192 Baidu EasyEdge Linux Development Kit 0.2.1(20190213)
2019-02-13 16:46:14,083 INFO [EasyEdge] [paddlev2_edge_predictor.cpp:60] 140606189016192 Allocate graph success.
2019-02-13 16:46:14,326 DEBUG [EasyEdge] [paddlev2_edge_predictor.cpp:143] 140606189016192 Inference costs 168 ms
1, 1:txt_frame, p:0.994905 loc: 0.168161, 0.153654, 0.920856, 0.779621
Done
```

### 2.2 预测视频流

```
./easyedge_video_inference {模型RES文件夹路径} {video_type} {video_src_path}
```

其中 video_type 支持三种: 

```
    video_type : 1                  // 本地视频文件
    video_type : 2                  // 摄像头的index
    video_type : 3                  // 网络视频流
```

video_src_path: 为 video_type 数值所对应的本地视频路径 、本地摄像头id、网络视频流地址，如：

```
    本地视频文件: ./easyedge_video_inference {模型RES文件夹路径} 1 ～/my_video_file.mp4
    本地摄像头: ./easyedge_video_inference {模型RES文件夹路径} 2 1 #/dev/video1
    网络视频流: ./easyedge_video_inference {模型RES文件夹路径} 3 rtmp://192.168.x.x:8733/live/src
```

注：以上路径是假模拟路径，开发者需要根据自己实际图像/视频，准备测试图像，并填写正确的测试路径。

# 预测API流程详解

本章节主要结合[2.测试Demo](#4)的Demo示例介绍推理API，方便开发者学习并将运行库嵌入到开发者的程序当中，更详细的API请参考`include/easyedge/easyedge*.h`文件。图像、视频的推理包含以下3个API，如代码step注释所示：

> ❗注意：<br>
> （1）`src`文件夹中包含完整可编译的cmake工程实例，建议开发者先行了解[cmake工程基本知识](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)。 <br>
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

* 接口
  
  ```cpp
  auto predictor = global_controller()->CreateEdgePredictor(config);
  predictor->init();
  ```

若返回非0，请查看输出日志排查错误原因。

## 3. 预测推理

### 3.1 预测图像

> 在Demo中展示了预测接口infer()传入cv::Mat& image图像内容，并将推理结果赋值给std::vector<EdgeResultData>& result。更多关于infer()的使用，可以根据参考`easyedge.h`头文件中的实际情况、参数说明自行传入需要的内容做推理

* 接口输入

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

* 接口返回
  
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

该字段返回了mask的游程编码，解析方式可参考 [http demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)

以上字段可以参考demo文件中使用opencv绘制的逻辑进行解析

### 3.2 预测视频

SDK 提供了支持摄像头读取、视频文件和网络视频流的解析工具类`VideoDecoding`，此类提供了获取视频帧数据的便利函数。通过`VideoConfig`结构体可以控制视频/摄像头的解析策略、抽帧策略、分辨率调整、结果视频存储等功能。对于抽取到的视频帧可以直接作为SDK infer 接口的参数进行预测。

* 接口输入

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

3.部分设备上的CSI摄像头尚未兼容，如遇到问题，可以通过工单、QQ交流群或微信交流群反馈。

具体接口调用流程，可以参考SDK中的`demo_video_inference`。

# FAQ

1. 如何处理一些 undefined reference / error while loading shared libraries?
   
   > 如：./easyedge_demo: error while loading shared libraries: libeasyedge.so.1: cannot open shared object file: No such file or directory
   
    遇到该问题时，请找到具体的库的位置，设置LD_LIBRARY_PATH；或者安装缺少的库。
   
   > 示例一：libverify.so.1: cannot open shared object file: No such file or directory
   > 链接找不到libveirfy.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../lib 解决(实际冒号后面添加的路径以libverify.so文件所在的路径为准)
   
   > 示例二：libopencv_videoio.so.4.5: cannot open shared object file: No such file or directory
   >  链接找不到libopencv_videoio.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../thirdparty/opencv/lib 解决(实际冒号后面添加的路径以libopencv_videoio.so所在路径为准)
   
   > 示例三：GLIBCXX_X.X.X  not found
   > 链接无法找到glibc版本，请确保系统gcc版本>=SDK的gcc版本。升级gcc/glibc可以百度搜索相关文献。

2. 运行二进制时，提示 libverify.so cannot open shared object file
   
    可能cmake没有正确设置rpath, 可以设置LD_LIBRARY_PATH为sdk的lib文件夹后，再运行：
   
   ```bash
   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib ./easyedge_demo
   ```

3. 编译时报错：file format not recognized
   
    可能是因为在复制SDK时文件信息丢失。请将整个压缩包复制到目标设备中，再解压缩、编译。
