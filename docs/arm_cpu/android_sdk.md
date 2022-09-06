# 简介

本文档介绍FastDeploy中的模型SDK，在Android环境下：（1）推理操作步骤；（2）介绍模型SDK使用说明，方便开发者了解项目后二次开发。

<!--ts-->

* [简介](#简介)

* [系统支持说明](#系统支持说明)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. APP 标准版测试](#2-app-标准版测试)
    * [2.1 扫码体验](#21-扫码体验)
    * [2.2 源码运行](#22-源码运行)
  * [3. 精简版测试](#3-精简版测试)

* [SDK使用说明](#sdk使用说明)
  
  * [1.  集成指南](#1-集成指南)
    * [1.1 依赖库集成](#11-依赖库集成)
    * [1.2 添加权限](#12-添加权限)
    * [1.3 混淆规则（可选）](#13-混淆规则可选)
  * [2. API调用流程示例](#2-api调用流程示例)
    * [2.1 初始化](#21-初始化)
    * [2.2 预测图像](#22-预测图像)

* [错误码](#错误码)
  
  <!--te-->

# 系统支持说明

1. Android 版本支持范围：Android 5.0（API21）<= Android < Android 10（API 29）。

2. 硬件支持情况：支持 arm64-v8a 和 armeabi-v7a，暂不支持模拟器。 
* 官网测试机型：红米k30，Vivo v1981a，华为oxp-an00，华为cdy-an90，华为pct-al10，荣耀yal-al00，OPPO Reno5 Pro 5G
3. 其他说明
* 【图像分割类算法】（1）图像分割类算法，暂未提供实时摄像头推理功能，开发者可根据自己需要，进行安卓开发；（2）PP-Humanseg-Lite模型设计初衷为横屏视频会议等场景，本次安卓SDK仅支持竖屏场景，开发者可根据自己需要，开发横屏功能。
* 【OCR模型】OCR任务第一次启动任务，第一张推理时间久，属于正常情况（因为涉及到模型加载、预处理等工作）。

> 预测图像时运行内存不能过小，一般大于模型资源文件夹大小的3倍。

# 快速开始

## 1. 项目结构说明

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。SDK目录结构如下：

```
.EasyEdge-Android-SDK
├── app
│   ├── src/main
│   │   ├── assets
│   │   │   ├── demo                 
│   │   │   │   └── conf.json        # APP名字
│   │   │   ├── infer                # 模型资源文件夹，一套模型适配不同硬件、OS和部署方式
│   │   │   │   ├── model            # 模型结构文件 
│   │   │   │   ├── params           # 模型参数文件
│   │   │   │   ├── label_list.txt   # 模型标签文件
│   │   │   │   └── infer_cfg.json   # 模型前后处理等配置文件
│   │   ├── java/com.baidu.ai.edge/demo
│   │   │   ├── infertest                          # 通用ARM精简版测试
│   │   │   │   ├── TestInferClassifyTask.java     # 图像分类
│   │   │   │   ├── TestInferDetectionTask.java    # 物体检测
│   │   │   │   ├── TestInferSegmentTask.java      # 实例分割
│   │   │   │   ├── TestInferPoseTask.java         # 姿态估计
│   │   │   │   ├── TestInferOcrTask.java          # OCR
│   │   │   │   └── MainActivity.java              # 精简版启动 Activity
│   │   │   ├── MainActivity.java          # Demo APP 启动 Activity
│   │   │   ├── CameraActivity.java        # 摄像头UI逻辑
│   │   │   └── ...
│   │   └── ...
│   ├── libs
│   │   ├── armeabi-v7a            # v7a的依赖库
│   │   ├── arm64-v8a              # v8a的依赖库
│   │   └── easyedge-sdk.jar       # jar文件
│   └── ...
├── camera_ui    # UI模块，包含相机逻辑
├── README.md
└── ...          # 其他 gradle 等工程文件
```

## 2. APP 标准版测试

考虑部分Android开发板没有摄像头，因此本项目开发了标准版和精简版两种。标准版会调用Android系统的摄像头，采集摄像头来进行AI模型推理；精简版在没有摄像头的开发板上运行，需要开发者准备图像。开发者根据硬件情况，选择对应的版本。

### 2.1 扫码体验

扫描二维码（二维码见下载网页`体验Demo`），无需任何依赖，手机上下载即可直接体验。

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175854064-a31755d1-52b9-416d-b35d-885b7338a6cc.png" width="600"></div>

### 2.2 源码运行

（1）下载对应的SDK，解压工程。</br>
   <div align=center><img src="https://user-images.githubusercontent.com/54695910/175854071-f4c17de8-83c2-434e-882d-c175f4202a2d.png" width="600"></div>
（2）打开Android Studio， 点击 "Import Project..."，即：File->New-> "Import Project...", 选择解压后的目录。</br>
（3）手机链接Android Studio，并打开开发者模式。（不了解开发者模式的开发者，可浏览器搜索）</br>
（4）此时点击运行按钮，手机上会有新app安装完毕，运行效果和二维码扫描的一样。</br>

  <div align=center><img src="https://user-images.githubusercontent.com/54695910/175854049-988414c7-116a-4261-a0c7-2705cc199538.png" width="400"></div>

## 3. 精简版测试

* 考虑部分Android开发板没有摄像头，本项目提供了精简版本，精简版忽略摄像头等UI逻辑，可兼容如无摄像头的开发板测试。

* 精简版对应的测试图像路径，在代码`src/main/java/com.baidu.ai.edge/demo/TestInfer*.java`中进行了设置，开发者可以准备图像到对应路径测试，也可以修改java代码测试。

* 支持以下硬件环境的精简版测试：通用ARM：图像分类、物体检测、实例分割、姿态估计、文字识别。

示例代码位于 app 模块下 infertest 目录，修改 app/src/main/AndroidManifest.xml 中的启动 Activity 开启测试。
修改前：

```
<activity android:name=".MainActivity">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                infertest.MainActivity
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        <activity
            android:name=".CameraActivity"
            android:screenOrientation="portrait" >
        </activity>
```

修改后：      

```
<!-- 以通用ARM为例 -->
<activity android:name=".infertest.MainActivity">
    <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
    </intent-filter>
</activity>
```

注意：修改后，因为没有测试数据，需要开发者准备一张测试图像，放到 `app/src/main/asserts/` 路径下，并按照`app/src/main/java/com/baidu/ai/edge/demo/infertest/TestInfer*.java`中的图像命名要求对图像进行命名。

<div align="center">

| Demo APP 检测模型运行示例                                                                                   | 精简版检测模型运行示例                                                                                    |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| ![Demo APP](https://user-images.githubusercontent.com/54695910/175855181-595fd449-7351-4ec6-a3b8-68c021b152f6.jpeg) | ![精简版](https://user-images.githubusercontent.com/54695910/175855176-075f0c8a-b05d-4d60-a2a1-3f0204c6386e.jpeg) |
</div>

# SDK使用说明

本节介绍如何将 SDK 接入开发者的项目中使用。

## 1. 集成指南

步骤一：依赖库集成
步骤二：添加必要权限
步骤三：混淆配置（可选）

### 1.1 依赖库集成

A. 项目中未集成其他 jar 包和 so 文件：

```
// 1. 复制 app/libs 至项目的 app/libs 目录
// 2. 参考 app/build.gradle 配置 NDK 可用架构和 so 依赖库目录

android {
    ...
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
    sourceSets {
        main {
            jniLibs.srcDirs = ['libs']
        }
    }
}
```

B. 项目中已集成其他 jar 包，未集成 so 文件：

```
// 1. 复制 app/libs/easyedge-sdk.jar 与其他 jar 包同目录
// 2. 复制 app/libs 下 armeabi-v7a 和 arm64-v8a 目录至 app/src/main/jniLibs 目录下
// 3. 参考 app/build.gradle 配置 NDK 可用架构

android {
    ...
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

C. 项目中已集成其他 jar 包和 so 文件：

```
// 1. 复制 app/libs/easyedge-sdk.jar 与其他 jar 包同目录
// 2. 融合 app/libs 下 armeabi-v7a 和 arm64-v8a 下的 so 文件与其他同架构 so 文件同目录
// 3. 参考 app/build.gradle 配置 NDK 可用架构

android {
    ...
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'    // 只支持 v7a 和 v8a 两种架构，有其他架构需删除
        }
    }
}
```

### 1.2 添加权限

参考 app/src/main/AndroidManifest.xml 中配置的权限。

```
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
```

### 1.3 混淆规则（可选）

请不要混淆 jar 包文件，参考 app/proguard-rules.pro 配置。

```
-keep class com.baidu.ai.edge.core.*.*{ *; }
```

## 2. API调用流程示例

以通用ARM的图像分类预测流程为例，详细说明请参考后续章节：

```
try {
    // step 1-1: 准备配置类
    InferConfig config = new InferConfig(context.getAssets(), "infer");

    // step 1-2: 准备预测 Manager
    InferManager manager = new InferManager(context, config, "");

    // step 2-1: 准备待预测的图像，必须为 Bitmap.Config.ARGB_8888 格式，一般为默认格式
    Bitmap image = getFromSomeWhere();

    // step 2-2: 预测图像
    List<ClassificationResultModel> results = manager.classify(image, 0.3f);

    // step 3: 解析结果
    for (ClassificationResultModel resultModel : results) {
        Log.i(TAG, "labelIndex=" + resultModel.getLabelIndex() 
                + ", labelName=" + resultModel.getLabel() 
                + ", confidence=" + resultModel.getConfidence());
    }

    // step 4: 释放资源。预测完毕请及时释放资源
    manager.destroy();
} catch (Exception e) {
    Log.e(TAG, e.getMessage());
}
```

### 2.1 初始化

**准备配置类**
芯片与配置类对应关系：

- 通用ARM：InferConfig

```
// 示例
// 参数二为芯片对应的模型资源文件夹名称
InferConfig config = new InferConfig(context.getAssets(), "infer");
```

**准备预测 Manager**
芯片与 Manager 对应关系：

- 通用ARM：InferManager

```
// 示例
// 参数二为配置类对象
// 参数三保持空字符串即可
InferManager manager = new InferManager(context, config, "");
```

>  **注意**
> 
> 1. 同一时刻只能有且唯一有效的 Manager，若要新建一个 Manager，之前创建的 Manager 需先调用 destroy() 销毁；
> 2. Manager 的任何方法都不能在 UI 线程调用；
> 3. Manager 的任何成员变量及方法由于线程同步问题，都必须在同一个线程中执行；

### 2.2 预测图像

本节介绍各种模型类型的预测函数及结果解析。

> **注意**
> 预测函数可以多次调用，但必须在同一个线程中，不支持并发
> 预测函数中的 confidence 非必需，默认使用模型推荐值。填 0 可返回所有结果
> 待预测的图像必须为 Bitmap.Config.ARGB_8888 格式的 Bitmap

**图像分类**

```
// 预测函数
List<ClassificationResultModel> classify(Bitmap bitmap) throws BaseException;
List<ClassificationResultModel> classify(Bitmap bitmap, float confidence) throws BaseException;

// 返回结果
ClassificationResultModel
- label: 分类标签，定义在label_list.txt中
- labelIndex: 分类标签对应的序号
- confidence: 置信度，0-1
```

**物体检测**

```
// 预测函数
List<DetectionResultModel> detect(Bitmap bitmap) throws BaseException;
List<DetectionResultModel> detect(Bitmap bitmap, float confidence) throws BaseException;

// 返回结果
DetectionResultModel
- label: 标签，定义在label_list.txt中
- confidence: 置信度，0-1
- bounds: Rect，包含左上角和右下角坐标，指示物体在图像中的位置
```

**实例分割**

```
// 预测函数
List<SegmentationResultModel> segment(Bitmap bitmap) throws BaseException;
List<SegmentationResultModel> segment(Bitmap bitmap, float confidence) throws BaseException;

// 返回结果
SegmentationResultModel
- label: 标签，定义在label_list.txt中
- confidence: 置信度，0-1
- lableIndex: 标签对应的序号
- box: Rect，指示物体在图像中的位置
- mask: byte[]，表示原图大小的0，1掩码，绘制1的像素即可得到当前对象区域
- maskLEcode: mask的游程编码
```

> 关于 maskLEcode 的解析方式可参考 [http demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)

**姿态估计**

```
// 预测函数
List<PoseResultModel> pose(Bitmap bitmap) throws BaseException;

// 返回结果
PoseResultModel
- label: 标签，定义在label_list.txt中
- confidence: 置信度，0-1
- points: Pair<Point, Point>, 2个点构成一条线
```

**文字识别**

```
// 预测函数
List<OcrResultModel> ocr(Bitmap bitmap) throws BaseException;
List<OcrResultModel> ocr(Bitmap bitmap, float confidence) throws BaseException;

// 返回结果
OcrResultModel
- label: 识别出的文字
- confidence: 置信度，0-1
- points: List<Point>, 文字所在区域的点位
```

# 错误码

| 错误码  | 错误描述                           | 详细描述及解决方法                                                                            |
| ---- | ------------------------------ | ------------------------------------------------------------------------------------ |
| 1001 | assets 目录下用户指定的配置文件不存在         | SDK可以使用assets目录下config.json作为配置文件。如果传入的config.json不在assets目录下，则有此报错                  |
| 1002 | 用户传入的配置文件作为json解析格式不准确，如缺少某些字段 | 正常情况下，demo中的config.json不要修改                                                          |
| 19xx | Sdk内部错误                        | 请与百度人员联系                                                                             |
| 2001 | XxxxMANAGER 只允许一个实例            | 如已有XxxxMANAGER对象，请调用destory方法                                                        |
| 2002 | XxxxMANAGER  已经调用过destory方法    | 在一个已经调用destory方法的DETECT_MANAGER对象上，不允许再调用任何方法                                        |
| 2003 | 传入的assets下模型文件路径为null          | XxxxConfig.getModelFileAssetPath() 返回为null。由setModelFileAssetPath(null）导致            |
| 2011 | libedge-xxxx.so 加载失败           | System.loadLibrary("edge-xxxx"); libedge-xxxx.so 没有在apk中。CPU架构仅支持armeabi-v7a arm-v8a |
| 2012 | JNI内存错误                        | heap的内存不够                                                                            |
| 2103 | license过期                      | license失效或者系统时间有异常                                                                   |
| 2601 | assets 目录下模型文件打开失败             | 请根据报错信息检查模型文件是否存在                                                                    |
| 2611 | 检测图片时，传递至引擎的图片二进制与长宽不符合        | 具体见报错信息                                                                              |
| 27xx | Sdk内部错误                        | 请与百度人员联系                                                                             |
| 28xx | 引擎内部错误                         | 请与百度人员联系                                                                             |
| 29xx | Sdk内部错误                        | 请与百度人员联系                                                                             |
| 3000 | so加载错误                         | 请确认所有so文件存在于apk中                                                                     |
| 3001 | 模型加载错误                         | 请确认模型放置于能被加载到的合法路径中，并确保config.json配置正确                                               |
| 3002 | 模型卸载错误                         | 请与百度人员联系                                                                             |
| 3003 | 调用模型错误                         | 在模型未加载正确或者so库未加载正确的情况下调用了分类接口                                                        |
| 50xx | 在线模式调用异常                       | 请与百度人员联系                                                                             |
