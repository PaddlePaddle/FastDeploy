[English](README.md) | 简体中文
# 目标检测 SCRFD Android Demo 使用文档  

在 Android 上实现实时的人脸检测功能，此 Demo 有很好的的易用性和开放性，如在 Demo 中跑自己训练好的模型等。

## 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

## 部署步骤

1. 目标检测 SCRFD Demo 位于 `fastdeploy/examples/vision/facedet/scrfd/android` 目录
2. 用 Android Studio 打开 scrfd/android 工程
3. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **注意：**
>> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod SDK location` 为您本机配置的 SDK 所在路径。

4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载预编译的 FastDeploy Android 库 以及 模型文件，需要联网)
成功后效果如下，图一：APP 安装到手机；图二： APP 打开后的效果，会自动识别图片中的人脸并绘制框；图三：APP设置选项，点击右上角的设置图片，可以设置不同选项进行体验。

 | APP 图标 | APP 效果 | APP设置项
  | ---     | --- | --- |
  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg">  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203261714-c74631dd-ec5b-4738-81a3-8dfc496f7547.gif"> | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg"> |  

## SCRFD Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PaddleSegModel初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  

```java
// 构造函数: constructor w/o label file
public SCRFD(); // 空构造函数，之后可以调用init初始化
public SCRFD(String modelFile, String paramsFile);
public SCRFD(String modelFile, String paramsFile, RuntimeOption option);
// 手动调用init初始化: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, RuntimeOption option);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap)；
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, float confThreshold, float nmsIouThreshold)； // 设置置信度阈值和NMS阈值
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float confThreshold, float nmsIouThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float confThreshold, float nmsIouThreshold); // 只渲染 不保存图片
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

- RuntimeOption设置说明  
```java  
public void enableLiteFp16(); // 开启fp16精度推理
public void disableLiteFP16(); // 关闭fp16精度推理
public void setCpuThreadNum(int threadNum); // 设置线程数
public void setLitePowerMode(LitePowerMode mode);  // 设置能耗模式
public void setLitePowerMode(String modeStr);  // 通过字符串形式设置能耗模式
```

- 模型结果SegmentationResult说明  
```java
public class FaceDetectionResult {
  public float[][] mBoxes; // [n,4] 检测框 (x1,y1,x2,y2)
  public float[] mScores;  // [n]   每个检测框得分(置信度，概率值)
  public float[][] mLandmarks; // [nx?,2] 每个检测到的人脸对应关键点
  int mLandmarksPerFace = 0;  // 每个人脸对应的关键点个数
  public boolean initialized(); // 检测结果是否有效
}  
```  
其他参考：C++/Python对应的FaceDetectionResult说明: [api/vision_results/face_detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/face_detection_result.md)


- 模型调用示例1：使用构造函数以及默认的RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.FaceDetectionResult;
import com.baidu.paddle.fastdeploy.vision.facedet.SCRFD;

// 初始化模型
SCRFD model = new SCRFD(
  "scrfd_500m_bnkps_shape320x320_pd/model.pdmodel",
  "scrfd_500m_bnkps_shape320x320_pd/model.pdiparams");

// 读取图片: 以下仅为读取Bitmap的伪代码
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// 直接预测返回 FaceDetectionResult
FaceDetectionResult result = model.predict(ARGB8888ImageBitmap);

// 释放模型资源  
model.release();
```  

- 模型调用示例2: 在合适的程序节点，手动调用init，并自定义RuntimeOption
```java  
// import 同上 ...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.FaceDetectionResult;
import com.baidu.paddle.fastdeploy.vision.facedet.SCRFD;
// 新建空模型
SCRFD model = new SCRFD();  
// 模型路径
String modelFile = "scrfd_500m_bnkps_shape320x320_pd/model.pdmodel";
String paramFile = "scrfd_500m_bnkps_shape320x320_pd/model.pdiparams";
// 指定RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();  
// 使用init函数初始化
model.init(modelFile, paramFile, option);
// Bitmap读取、模型预测、资源释放 同上 ...
```
更详细的用法请参考 [FaceDetMainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/facedet/FaceDetMainActivity.java) 中的用法

## 替换 FastDeploy SDK和模型  
替换FastDeploy预测库和模型的步骤非常简单。预测库所在的位置为 `app/libs/fastdeploy-android-sdk-xxx.aar`，其中 `xxx` 表示当前您使用的预测库版本号。模型所在的位置为，`app/src/main/assets/models/scrfd_500m_bnkps_shape320x320_pd`。  
- 替换FastDeploy Android SDK: 下载或编译最新的FastDeploy Android SDK，解压缩后放在 `app/libs` 目录下；详细配置文档可参考:  
     - [在 Android 中使用 FastDeploy Java SDK](../../../../../java/android/)

- 替换SCRFD模型的步骤：  
  - 通过X2Paddle导出其他版本的SCRFD模型，请参考 [SCRFD文档](../README.md) 以及 [X2Paddle](https://github.com/PaddlePaddle/X2Paddle)  
  - 将您的SCRFD模型放在 `app/src/main/assets/models` 目录下；  
  - 修改 `app/src/main/res/values/strings.xml` 中模型路径的默认值，如：  
```xml
<!-- 将这个路径指修改成您的模型，如 models/scrfd_500m_bnkps_shape320x320_pd -->
<string name="FACEDET_MODEL_DIR_DEFAULT">models/scrfd_500m_bnkps_shape320x320_pd</string>  
```  

## 更多参考文档
如果您想知道更多的FastDeploy Java API文档以及如何通过JNI来接入FastDeploy C++ API感兴趣，可以参考以下内容:  
- [在 Android 中使用 FastDeploy Java SDK](../../../../../java/android/)
- [在 Android 中使用 FastDeploy C++ SDK](../../../../../docs/cn/faq/use_cpp_sdk_on_android.md)  
