# FastDeploy Android AAR 包使用文档  
FastDeploy Android SDK 目前支持图像分类、目标检测、OCR文字识别、语义分割和人脸检测等任务，对更多的AI任务支持将会陆续添加进来。以下为各个任务对应的API文档，在Android下使用FastDeploy中集成的模型，只需以下几个步骤：  
- 模型初始化  
- 调用`predict`接口  
- 可视化验证

## 图像分类
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

## 目标检测
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

## OCR文字识别  
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

## 语义分割  
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
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## 人脸检测  
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
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // 只渲染 不保存图片
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
// 预测并且可视化：预测结果以及可视化，并将可视化后的图片保存到指定的途径，以及将可视化结果渲染在Bitmap上
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // 只渲染 不保存图片
```
- 模型资源释放 API：调用 release() API 可以释放模型资源，返回true表示释放成功，false表示失败；调用 initialized() 可以判断模型是否初始化成功，true表示初始化成功，false表示失败。
```java
public boolean release(); // 释放native资源  
public boolean initialized(); // 检查是否初始化成功
```

## 结果说明  
- 图像分类ClassifyResult说明  
```java
public class ClassifyResult {
  public float[] mScores;  // [n]   得分
  public int[] mLabelIds;  // [n]   分类ID
  public boolean initialized(); // 检测结果是否有效
}
```  

- 目标检测DetectionResult说明  
```java
public class DetectionResult {
  public float[][] mBoxes; // [n,4] 检测框 (x1,y1,x2,y2)
  public float[] mScores;  // [n]   得分
  public int[] mLabelIds;  // [n]   分类ID
  public boolean initialized(); // 检测结果是否有效
}
```  

- OCR文字识别OCRResult说明  
```java
public class OCRResult {
  public int[][] mBoxes;  // [n,8]
  public String[] mText;  // [n]
  public float[] mRecScores;  // [n]
  public float[] mClsScores;  // [n]
  public int[] mClsLabels;  // [n]
  public boolean mInitialized = false;
  public boolean initialized(); // 检测结果是否有效
}
```  

- 语义分割SegmentationResult结果说明  
```java
public class SegmentationResult {
  public int[] mLabelMap;  //  预测到的label map 每个像素位置对应一个label HxW
  public float[] mScoreMap; // 预测到的得分 map 每个像素位置对应一个score HxW
  public long[] mShape; // label map实际的shape (H,W)
  public boolean mContainScoreMap = false; // 是否包含 score map
  public boolean initialized(); // 检测结果是否有效
}  
```

- 人脸检测FaceDetectionResult结果说明  
```java
public class FaceDetectionResult {
  public float[][] mBoxes; // [n,4] 检测框 (x1,y1,x2,y2)
  public float[] mScores;  // [n]   得分
  public float[][] mLandmarks; // [nx?,2] 每个检测到的人脸对应关键点
  int mLandmarksPerFace = 0;  // 每个人脸对应的关键点个数
  public boolean initialized(); // 检测结果是否有效
}  
```

## RuntimeOption说明
- RuntimeOption设置说明  
```java
public class RuntimeOption {
  public void enableLiteFp16(); // 开启fp16精度推理
  public void disableLiteFP16(); // 关闭fp16精度推理
  public void setCpuThreadNum(int threadNum); // 设置线程数
  public void setLitePowerMode(LitePowerMode mode);  // 设置能耗模式
  public void setLitePowerMode(String modeStr);  // 通过字符串形式设置能耗模式
}
```

## 可视化接口  
FastDeploy Android SDK同时提供一些可视化接口，可用于快速验证推理结果。可视化API接口如下：

```java  
public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result);
public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result);
public static boolean visFaceDetection(Bitmap ARGB8888Bitmap, FaceDetectionResult result);
public static boolean visOcr(Bitmap ARGB8888Bitmap, OCRResult result);
public static boolean visSegmentation(Bitmap ARGB8888Bitmap, SegmentationResult result);
```

## 使用示例  
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
option.enableRecordTimeOfRuntime();
option.enableLiteFp16();
// 使用init函数初始化  
model.init(modelFile, paramFile, configFile, option);
// Bitmap读取、模型预测、资源释放 同上 ...
```
