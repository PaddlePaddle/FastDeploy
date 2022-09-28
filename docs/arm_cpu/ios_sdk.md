# 简介

本文档介绍FastDeploy中的模型SDK，在iOS环境下：（1）推理部署步骤；（2）介绍SDK使用说明，方便开发者了解项目后二次开发。

<!--ts-->

* [简介](#简介)

* [系统支持说明](#系统支持说明)
  
  * [1. 系统支持说明](#1-系统支持说明)
  * [2. SDK大小说明](#2-sdk大小说明)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. 测试Demo](#2-测试demo)

* [SDK使用说明](#sdk使用说明)
  
  * [1. 集成指南](#1-集成指南)
    * [1.1 依赖库集成](#11-依赖库集成)
  * [2. 调用流程示例](#2-调用流程示例)
    * [2.1 初始化](#21-初始化)
  * [2.2 预测图像](#22-预测图像)

* [FAQ](#faq)
  
  <!--te-->

# 系统支持说明

## 1. 系统支持说明

1. 系统支持：iOS 9.0及以上。

2. 硬件支持：支持 arm64 (Starndard architectures)，暂不支持模拟器。
   
   * 官方验证过的手机机型：大部分ARM 架构的手机、平板及开发板。

3.其他说明

    * 3.1 【图像分割类模型】（1）图像分割类Demo暂未提供实时摄像头录制拍摄的能力，开发者可根据自己需要，进行安卓开发完成；（2）PP-Humanseg-Lite模型设计初衷为横屏视频会议等场景，本次安卓开发仅支持述评场景，开发者可根据自己需要，开发横屏的Android功能。<br>
    
    * 3.2 【OCR模型】OCR任务第一次启动任务，第一张推理时间久，属于正常情况（因为涉及到模型加载、预处理等工作）。<br>

## 2. SDK大小说明

1. 模型资源文件大小影响 SDK 大小
2. SDK 包及 IPA 安装包虽然比较大，但最终安装到设备后所占大小会缩小很多。这与 multi architechtures、bitcode 和 AppStore 的优化有关。

# 快速开始

## 1. 项目结构说明

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GitHub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。SDK目录结构如下：

```
.EasyEdge-iOS-SDK
├── EasyDLDemo    # Demo工程文件
├── LIB            # 依赖库
├── RES
│   ├── easyedge         # 模型资源文件夹，一套模型适配不同硬件、OS和部署方式
│   ├── conf.json        # Android、iOS系统APP名字需要
│   ├── model            # 模型结构文件 
│   ├── params           # 模型参数文件
│   ├── label_list.txt   # 模型标签文件
│   ├── infer_cfg.json   # 模型前后处理等配置文件
└── DOC            # 文档
```

## 2. 测试Demo

按如下步骤可直接运行 SDK 体验 Demo：  
步骤一：用 Xcode 打开 `EasyDLDemo/EasyDLDemo.xcodeproj`  
步骤二：配置开发者自己的签名（不了解签名机制的，可以看FAQ [iOS签名介绍](#100)）</br>
步骤三：连接手机运行，不支持模拟器  

检测模型运行示例：

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175854078-4f1f761d-0629-411a-92cc-6f4180164ca5.png" width="400"></div>

# SDK使用说明

本节介绍如何将 SDK 接入开发者的项目中使用。

## 1. 集成指南

步骤一：依赖库集成
步骤二：`import <EasyDL/EasyDL.h>`

### 1.1 依赖库集成

1. 复制 LIB 目录至项目合适的位置
2. 配置 Build Settings 中 Search paths: 以 SDK 中 LIB 目录路径为例
- Framework Search Paths：`${PROJECT_DIR}/../LIB/lib`
- Header Search Paths：`${PROJECT_DIR}/../LIB/include`
- Library Search Paths：`${PROJECT_DIR}/../LIB/lib`

> 集成过程如出现错误，请参考 Demo 工程对依赖库的引用

## 2. 调用流程示例

以通用ARM的图像分类预测流程为例，详细说明请参考后续章节：

```
NSError *err;

// step 1: 初始化模型
EasyDLModel *model = [[EasyDLModel alloc] initModelFromResourceDirectory:@"easyedge" withError:&err];

// step 2: 准备待预测的图像
UIImage *image = ...;

// step 3: 预测图像
NSArray *results = [model detectUIImage:image withFilterScore:0 andError:&err];

// step 4: 解析结果
for (id res in results) {
    EasyDLClassfiData *clsData = (EasyDLClassfiData *) res;
    NSLog(@"labelIndex=%d, labelName=%@, confidence=%f", clsData.category, clsData.label, clsData.accuracy);
}
```

### 2.1 初始化

```
// 示例
// 参数一为模型资源文件夹名称
EasyDLModel *model = [[EasyDLModel alloc] initModelFromResourceDirectory:@"easyedge" withError:&err];
```

> 模型资源文件夹需以 folder reference 方式加入 Xcode 工程，如 `RES/easyedge` 文件夹在 Demo 工程中表现为蓝色

### 2.2 预测图像

所有模型类型通过以下接口获取预测结果：

```
// 返回的数组类型不定
NSArray *results = [model detectUIImage:image withFilterScore:0 andError:&err];
```

返回的数组类型如下，具体可参考 `EasyDLResultData.h` 中的定义：
| 模型类型 | 类型 |
| --- | ---- |
| 图像分类   | EasyDLClassfiData |
| 物体检测/人脸检测   | EasyDLObjectDetectionData |
| 实例分割   | EasyDLObjSegmentationData |
| 姿态估计   | EasyDLPoseData |
| 文字识别   | EasyDLOcrData |

# FAQ

1. 如何多线程并发预测？

SDK内部已经能充分利用多核的计算能力。不建议使用并发来预测。

如果开发者想并发使用，请务必注意`EasyDLModel`所有的方法都不是线程安全的。请初始化多个实例进行并发使用，如

```c
- (void)testMultiThread {
    UIImage *img = [UIImage imageNamed:@"1.jpeg"];
    NSError *err;
    EasyDLModel * model1 = [[EasyDLModel alloc] initModelFromResourceDirectory:@"easyedge" withError:&err];
    EasyDLModel * model2 = [[EasyDLModel alloc] initModelFromResourceDirectory:@"easyedge" withError:&err];

    dispatch_queue_t queue1 = dispatch_queue_create("testQueue", DISPATCH_QUEUE_CONCURRENT);
    dispatch_queue_t queue2 = dispatch_queue_create("testQueue2", DISPATCH_QUEUE_CONCURRENT);

    dispatch_async(queue1, ^{
        NSError *detectErr;
        for(int i = 0; i < 1000; ++i) {
            NSArray * res = [model1 detectUIImage:img withFilterScore:0 andError:&detectErr];
            NSLog(@"1: %@", res[0]);
        }
    });

    dispatch_async(queue2, ^{
        NSError *detectErr;
        for(int i = 0; i < 1000; ++i) {
            NSArray * res = [model2 detectUIImage:img withFilterScore:0 andError:&detectErr];
            NSLog(@"2: %@", res[0]);
        }
    });
}
```

2. 编译时出现 Undefined symbols for architecture arm64: ...
* 出现 `cxx11, vtable` 字样：请引入 `libc++.tbd`
* 出现 `cv::Mat` 字样：请引入 `opencv2.framework`
* 出现 `CoreML`, `VNRequest` 字样：请引入`CoreML.framework` 并务必`#import <CoreML/CoreML.h> ` 
3. 运行时报错 Image not found: xxx ...

请Embed具体报错的库。

4. 编译时报错：Invalid bitcode version

这个可能是开发者使用的 Xcode 低于12导致，可以升级至12版本。

5. 错误说明

SDK 的方法会返回 NSError，直接返回的 NSError 的错误码定义在 `EasyDLDefine.h - EEasyDLErrorCode` 中。NSError 附带 message （有时候会附带 NSUnderlyingError），开发者可根据 code 和 message 进行错误判断和处理。

6. iOS签名说明

iOS 签名是苹果生态对 APP 开发者做的限定，对于个人开发者是免费的，对于企业开发者（譬如APP要上架应用市场），是收费的。此处，仅简单说明作为普通开发者，第一次尝试使用 Xcode编译代码，需要进行的签名操作。<br>
（1）在Xcode/Preferences/Accounts 中添加个人Apple ID;<br>
（2）在对应的EasyDLDemo中做如下图设置：<br>

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175854089-aa1d1af8-7daa-43ae-868d-32041c27ad86.jpg" width="600"></div>
（3）（2）后会在手机上安装好对应APP，还需要在手机上`设置/通用/设备管理/开发者应用/信任appleID`，才能运行该 APP。
