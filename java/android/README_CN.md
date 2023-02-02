简体中文 | [English](README.md)

# FastDeploy Android AAR 包使用文档  
FastDeploy Android SDK 目前支持图像分类、目标检测、OCR文字识别、语义分割和人脸检测等任务，对更多的AI任务支持将会陆续添加进来。以下为各个任务对应的API文档，在Android下使用FastDeploy中集成的模型，只需以下几个步骤：  
- 模型初始化  
- 调用`predict`接口  
- 可视化验证（可选）

|图像分类|目标检测|OCR文字识别|人像分割|人脸检测|  
|:---:|:---:|:---:|:---:|:---:|  
|![classify](https://user-images.githubusercontent.com/31974251/203261658-600bcb09-282b-4cd3-a2f2-2c733a223b03.gif)|![detection](https://user-images.githubusercontent.com/31974251/203261763-a7513df7-e0ab-42e5-ad50-79ed7e8c8cd2.gif)|![ocr](https://user-images.githubusercontent.com/31974251/203261817-92cc4fcd-463e-4052-910c-040d586ff4e7.gif)|![seg](https://user-images.githubusercontent.com/31974251/203267867-7c51b695-65e6-402e-9826-5d6d5864da87.gif)|![face](https://user-images.githubusercontent.com/31974251/203261714-c74631dd-ec5b-4738-81a3-8dfc496f7547.gif)|

## 内容目录

- [下载及配置SDK](#SDK)
- [图像分类API](#Classification)  
- [目标检测API](#Detection)  
- [语义分割API](#Segmentation)  
- [OCR文字识别API](#OCR)  
- [人脸检测API](#FaceDetection)  
- [识别结果说明](#VisionResults)
- [RuntimeOption说明](#RuntimeOption)  
- [可视化接口API](#Visualize)
- [模型使用示例](#Demo)
- [App示例工程使用方式](#App)  

## 下载及配置SDK  
<div id="SDK"></div>  

### 下载 FastDeploy Android SDK  
Release版本（Java SDK 目前仅支持Android）  

| 平台 | 文件 | 说明 |
| :--- | :--- | :---- |
| Android Java SDK | [fastdeploy-android-sdk-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-0.0.0.aar) | NDK 20 编译产出, minSdkVersion 15,targetSdkVersion 28 |

更多预编译库信息，请参考: [download_prebuilt_libraries.md](../../docs/cn/build_and_install/download_prebuilt_libraries.md)

### 配置 FastDeploy Android SDK  

首先，将fastdeploy-android-sdk-xxx.aar拷贝到您Android工程的libs目录下，其中`xxx`表示您所下载的SDK的版本号。
```shell
├── build.gradle
├── libs
│   └── fastdeploy-android-sdk-xxx.aar
├── proguard-rules.pro
└── src
```

然后，在您的Android工程中的build.gradble引入FastDeploy SDK，如下：
```java  
dependencies {
    implementation fileTree(include: ['*.aar'], dir: 'libs')
    implementation 'com.android.support:appcompat-v7:28.0.0'
    // ...
}
```

## 图像分类API  

<div id="Classification"></div>  

### PaddleClasModel Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PaddleClasModel初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - configFile: String, 模型推理的预处理配置文件，如 infer_cfg.yml  
  - labelFile: String, 可选参数，表示label标签文件所在路径，用于可视化，如 imagenet1k_label_list.txt，每一行包含一个label  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  


```java
// 构造函数: constructor w/o label file
public PaddleClasModel(); // 空构造函数，之后可以调用init初始化
public PaddleClasModel(String modelFile, String paramsFile, String configFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// 手动调用init初始化: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public ClassifyResult predict(Bitmap ARGB8888Bitmap)；
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public ClassifyResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public ClassifyResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // 只渲染 不保存图片
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## 目标检测API

<div id="Detection"></div>  

### PicoDet Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PicoDet初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - configFile: String, 模型推理的预处理配置文件，如 infer_cfg.yml  
  - labelFile: String, 可选参数，表示label标签文件所在路径，用于可视化，如 coco_label_list.txt，每一行包含一个label  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  

```java
// 构造函数: constructor w/o label file
public PicoDet(); // 空构造函数，之后可以调用init初始化
public PicoDet(String modelFile, String paramsFile, String configFile);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile);
public PicoDet(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// 手动调用init初始化: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public DetectionResult predict(Bitmap ARGB8888Bitmap)；
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public DetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public DetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // 只渲染 不保存图片
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## OCR文字识别API  

<div id="OCR"></div>  

### PP-OCRv2 & PP-OCRv3 Java API 说明
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。 PP-OCR初始化参数说明如下：
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - labelFile: String, 可选参数，表示label标签文件所在路径，用于可视化，如 ppocr_keys_v1.txt，每一行包含一个label  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。
与其他模型不同的是，PP-OCRv2 和 PP-OCRv3 包含 DBDetector、Classifier和Recognizer等基础模型，以及PPOCRv2和PPOCRv3等pipeline类型。  
```java
// 构造函数: constructor w/o label file
public DBDetector(String modelFile, String paramsFile);
public DBDetector(String modelFile, String paramsFile, RuntimeOption option);
public Classifier(String modelFile, String paramsFile);
public Classifier(String modelFile, String paramsFile, RuntimeOption option);
public Recognizer(String modelFile, String paramsFile, String labelPath);
public Recognizer(String modelFile, String paramsFile,  String labelPath, RuntimeOption option);
public PPOCRv2();  // 空构造函数，之后可以调用init初始化
// Constructor w/o classifier
public PPOCRv2(DBDetector detModel, Recognizer recModel);
public PPOCRv2(DBDetector detModel, Classifier clsModel, Recognizer recModel);
public PPOCRv3();  // 空构造函数，之后可以调用init初始化
// Constructor w/o classifier
public PPOCRv3(DBDetector detModel, Recognizer recModel);
public PPOCRv3(DBDetector detModel, Classifier clsModel, Recognizer recModel);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public OCRResult predict(Bitmap ARGB8888Bitmap)；
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public OCRResult predict(Bitmap ARGB8888Bitmap, String savedImagePath);
public OCRResult predict(Bitmap ARGB8888Bitmap, boolean rendering); // 只渲染 不保存图片
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## 语义分割API  

<div id="Segmentation"></div>  

### PaddleSegModel Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PaddleSegModel初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - configFile: String, 模型推理的预处理配置文件，如 infer_cfg.yml  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  

```java
// 构造函数: constructor w/o label file
public PaddleSegModel(); // 空构造函数，之后可以调用init初始化
public PaddleSegModel(String modelFile, String paramsFile, String configFile);
public PaddleSegModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
// 手动调用init初始化: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
```  
- 模型预测 API：模型预测API包含直接预测的API以及带可视化功能的API。直接预测是指，不保存图片以及不渲染结果到Bitmap上，仅预测推理结果。预测并且可视化是指，预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap(目前支持ARGB8888格式的Bitmap), 后续可将该Bitmap在camera中进行显示。
```java
// 直接预测：不保存图片以及不渲染结果到Bitmap上
public SegmentationResult predict(Bitmap ARGB8888Bitmap)；
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public SegmentationResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float weight);
public SegmentationResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float weight); // 只渲染 不保存图片
// 修改result，而非返回result，关注性能的用户可以将以下接口与SegmentationResult的CxxBuffer一起使用
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result)；
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, String savedImagePath, float weight);
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, boolean rendering, float weight);
```
- 设置竖屏或横屏模式: 对于 PP-HumanSeg系列模型，必须要调用该方法设置竖屏模式为true.
```java  
public void setVerticalScreenFlag(boolean flag);
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## 人脸检测API  

<div id="FaceDetection"></div>  

### SCRFD Java API 说明  
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

### YOLOv5Face Java API 说明  
- 模型初始化 API: 模型初始化API包含两种方式，方式一是通过构造函数直接初始化；方式二是，通过调用init函数，在合适的程序节点进行初始化。PaddleSegModel初始化参数说明如下：  
  - modelFile: String, paddle格式的模型文件路径，如 model.pdmodel
  - paramFile: String, paddle格式的参数文件路径，如 model.pdiparams  
  - option: RuntimeOption，可选参数，模型初始化option。如果不传入该参数则会使用默认的运行时选项。  

```java
// 构造函数: constructor w/o label file
public YOLOv5Face(); // 空构造函数，之后可以调用init初始化
public YOLOv5Face(String modelFile, String paramsFile);
public YOLOv5Face(String modelFile, String paramsFile, RuntimeOption option);
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

## 识别结果说明  

<div id="VisionResults"></div>  

- 图像分类ClassifyResult说明  
```java
public class ClassifyResult {
  public float[] mScores;  // [n]   每个类别的得分(概率)
  public int[] mLabelIds;  // [n]   分类ID 具体的类别类型
  public boolean initialized(); // 检测结果是否有效
}
```  
其他参考：C++/Python对应的ClassifyResult说明: [api/vision_results/classification_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result.md)

- 目标检测DetectionResult说明  
```java
public class DetectionResult {
  public float[][] mBoxes; // [n,4] 检测框 (x1,y1,x2,y2)
  public float[] mScores;  // [n]   每个检测框得分(置信度，概率值)
  public int[] mLabelIds;  // [n]   分类ID
  public boolean initialized(); // 检测结果是否有效
}
```  
其他参考：C++/Python对应的DetectionResult说明: [api/vision_results/detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/detection_result.md)

- OCR文字识别OCRResult说明  
```java
public class OCRResult {
  public int[][] mBoxes;  // [n,8] 表示单张图片检测出来的所有目标框坐标 每个框以8个int数值依次表示框的4个坐标点，顺序为左下，右下，右上，左上
  public String[] mText;  // [n] 表示多个文本框内被识别出来的文本内容
  public float[] mRecScores;  // [n] 表示文本框内识别出来的文本的置信度
  public float[] mClsScores;  // [n] 表示文本框的分类结果的置信度
  public int[] mClsLabels;  // [n] 表示文本框的方向分类类别
  public boolean initialized(); // 检测结果是否有效
}
```  
其他参考：C++/Python对应的OCRResult说明: [api/vision_results/ocr_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/ocr_result.md)

- 语义分割SegmentationResult结果说明  
```java
public class SegmentationResult {
  public int[] mLabelMap;  //  预测到的label map 每个像素位置对应一个label HxW
  public float[] mScoreMap; // 预测到的得分 map 每个像素位置对应一个score HxW
  public long[] mShape; // label map实际的shape (H,W)
  public boolean mContainScoreMap = false; // 是否包含 score map
  // 用户可以选择直接使用CxxBuffer，而非通过JNI拷贝到Java层，
  // 该方式可以一定程度上提升性能
  public void setCxxBufferFlag(boolean flag); // 设置是否为CxxBuffer模式
  public boolean releaseCxxBuffer(); // 手动释放CxxBuffer!!!
  public boolean initialized(); // 检测结果是否有效
}  
```
其他参考：C++/Python对应的SegmentationResult说明: [api/vision_results/segmentation_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/segmentation_result.md)

- 人脸检测FaceDetectionResult结果说明  
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

## RuntimeOption说明  

<div id="RuntimeOption"></div>  

- RuntimeOption设置说明  
```java
public class RuntimeOption {
  public void enableLiteFp16(); // 开启fp16精度推理
  public void disableLiteFP16(); // 关闭fp16精度推理
  public void enableLiteInt8(); // 开启int8精度推理，针对量化模型
  public void disableLiteInt8(); // 关闭int8精度推理
  public void setCpuThreadNum(int threadNum); // 设置线程数
  public void setLitePowerMode(LitePowerMode mode);  // 设置能耗模式
  public void setLitePowerMode(String modeStr);  // 通过字符串形式设置能耗模式
}
```

## 可视化接口  

<div id="Visualize"></div>  

FastDeploy Android SDK同时提供一些可视化接口，可用于快速验证推理结果。以下接口均把结果result渲染在输入的Bitmap上。具体的可视化API接口如下：

```java  
public class Visualize {
  // 默认参数接口
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result);
  public static boolean visFaceDetection(Bitmap ARGB8888Bitmap, FaceDetectionResult result);
  public static boolean visOcr(Bitmap ARGB8888Bitmap, OCRResult result);
  public static boolean visSegmentation(Bitmap ARGB8888Bitmap, SegmentationResult result);
  // 有可设置参数的可视化接口  
  // visDetection: 可设置阈值（大于该阈值的框进行绘制）、框线大小、字体大小、类别labels等
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, float scoreThreshold);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, float scoreThreshold, int lineSize, float fontSize);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, String[] labels);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, String[] labels, float scoreThreshold, int lineSize, float fontSize);
  // visClassification: 可设置阈值（大于该阈值的框进行绘制）、字体大小、类别labels等
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, float scoreThreshold,float fontSize);
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, String[] labels);
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, String[] labels, float scoreThreshold,float fontSize);
  // visSegmentation: weight背景权重
  public static boolean visSegmentation(Bitmap ARGB8888Bitmap, SegmentationResult result, float weight);
  // visFaceDetection: 线大小、字体大小等
  public static boolean visFaceDetection(Bitmap ARGB8888Bitmap, FaceDetectionResult result, int lineSize, float fontSize);
}
```  
对应的可视化类型为：  
```java
import com.baidu.paddle.fastdeploy.vision.Visualize;
```

## 模型使用示例  

<div id="Demo"></div>  

- 模型调用示例1：使用构造函数以及默认的RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;

// 初始化模型
PicoDet model = new PicoDet("picodet_s_320_coco_lcnet/model.pdmodel",
                            "picodet_s_320_coco_lcnet/model.pdiparams",
                            "picodet_s_320_coco_lcnet/infer_cfg.yml");

// 模型推理
DetectionResult result = model.predict(ARGB8888ImageBitmap);  

// 释放模型资源  
model.release();
```  

- 模型调用示例2: 在合适的程序节点，手动调用init，并自定义RuntimeOption
```java  
// import 同上 ...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;
// 新建空模型
PicoDet model = new PicoDet();  
// 模型路径
String modelFile = "picodet_s_320_coco_lcnet/model.pdmodel";
String paramFile = "picodet_s_320_coco_lcnet/model.pdiparams";
String configFile = "picodet_s_320_coco_lcnet/infer_cfg.yml";
// 指定RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();
// 使用init函数初始化  
model.init(modelFile, paramFile, configFile, option);
// Bitmap读取、模型预测、资源释放 同上 ...
```

## App示例工程使用方式  
<div id="App"></div>  

FastDeploy在java/android/app目录下提供了一些示例工程，以下将介绍示例工程的使用方式。由于java/android目录下同时还包含JNI工程，因此想要使用示例工程的用户还需要配置NDK，如果您只关心Java API的使用，并且不想配置NDK，可以直接跳转到以下详细的案例链接。  

- [图像分类App示例工程](../../examples/vision/classification/paddleclas/android)  
- [目标检测App示例工程](../../examples/vision/detection/paddledetection/android)  
- [OCR文字识别App示例工程](../../examples/vision/ocr/PP-OCRv2/android)  
- [人像分割App示例工程](../../examples/vision/segmentation/paddleseg/android)  
- [人脸检测App示例工程](../../examples/vision/facedet/scrfd/android)  

### 环境准备

1. 在本地环境安装好 Android Studio 工具，详细安装方法请见[Android Stuido 官网](https://developer.android.com/studio)。
2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

**注意**：如果您的 Android Studio 尚未配置 NDK ，请根据 Android Studio 用户指南中的[安装及配置 NDK 和 CMake ](https://developer.android.com/studio/projects/install-ndk)内容，预先配置好 NDK 。您可以选择最新的 NDK 版本，或者使用 FastDeploy Android 预测库版本一样的 NDK

### 部署步骤

1. App示例工程位于 `fastdeploy/java/android/app` 目录
2. 用 Android Studio 打开 `fastdeploy/java/android` 工程，注意是`java/android`目录
3. 手机连接电脑，打开 USB 调试和文件传输模式，并在 Android Studio 上连接自己的手机设备（手机需要开启允许从 USB 安装软件权限）

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **注意：**
>> 如果您在导入项目、编译或者运行过程中遇到 NDK 配置错误的提示，请打开 ` File > Project Structure > SDK Location`，修改 `Andriod NDK location` 为您本机配置的 NDK 所在路径。本工程默认使用的NDK版本为20.
>> 如果您是通过 Andriod Studio 的 SDK Tools 下载的 NDK (见本章节"环境准备")，可以直接点击下拉框选择默认路径。
>> 还有一种 NDK 配置方法，你可以在 `java/android/local.properties` 文件中手动完成 NDK 路径配置，如下图所示
>> 如果以上步骤仍旧无法解决 NDK 配置错误，请尝试根据 Andriod Studio 官方文档中的[更新 Android Gradle 插件](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin)章节，尝试更新Android Gradle plugin版本。

4. 点击 Run 按钮，自动编译 APP 并安装到手机。(该过程会自动下载预编译的 FastDeploy Android 库 以及 模型文件，需要联网)
成功后效果如下，图一：APP 安装到手机；图二： APP 打开后的效果，会自动识别图片中的物体并标记；图三：APP设置选项，点击右上角的设置图片，可以设置不同选项进行体验。

  | APP 图标 | APP 效果 | APP设置项
  | ---     | --- | --- |
  | ![app_pic](https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg)   | ![app_res](https://user-images.githubusercontent.com/31974251/197169609-bb214af3-d6e7-4433-bb96-1225cddd441c.jpg) |  ![app_setup](https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg) |  

### 切换不同的场景  
App示例工程只需要在AndroidManifest.xml中切换不同的Activity即可编译不同场景的App进行体验。  

<p align="center">
<img width="788" alt="image" src="https://user-images.githubusercontent.com/31974251/203258255-b422d3e2-6004-465f-86b6-9fa61a27c6c2.png">
</p>  

- 图像分类场景  
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.baidu.paddle.fastdeploy.app.examples">
    <!-- ... -->
        <activity android:name=".classification.ClassificationMainActivity">
           <!--  -->
        </activity>
        <activity
            android:name=".classification.ClassificationSettingsActivity"
        </activity>
    </application>
</manifest>
```  
- 目标检测场景  
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.baidu.paddle.fastdeploy.app.examples">
    <!-- ... -->
        <activity android:name=".detection.DetectionMainActivity">
           <!--  -->
        </activity>
        <activity
            android:name=".detection.DetectionSettingsActivity"
        </activity>
    </application>
</manifest>
```  
- OCR文字识别场景  
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.baidu.paddle.fastdeploy.app.examples">
    <!-- ... -->
        <activity android:name=".ocr.OcrMainActivity">
            <!--  -->
        </activity>
        <activity
            android:name=".ocr.OcrSettingsActivity"
        </activity>
    </application>
</manifest>
```  
- 人像分割场景  
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.baidu.paddle.fastdeploy.app.examples">
    <!-- ... -->
        <activity android:name=".segmentation.SegmentationMainActivity">
            <!--  -->
        </activity>
        <activity
            android:name=".segmentation.SegmentationSettingsActivity"
        </activity>
    </application>
</manifest>
```  
- 人脸检测场景  
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.baidu.paddle.fastdeploy.app.examples">
    <!-- ... -->
        <activity android:name=".facedet.FaceDetMainActivity">
            <!--  -->
        </activity>
        <activity
            android:name=".facedet.FaceDetSettingsActivity"
        </activity>
    </application>
</manifest>
```
