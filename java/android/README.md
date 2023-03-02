English | [简体中文](README_CN.md)

# FastDeploy Android AAR Package  
Currently FastDeploy Android SDK supports image classification, object detection, OCR text recognition, semantic segmentation and face detection. More AI tasks will be added in the future. The following is the API documents for each task. To use the models integrated in FastDeploy on Android, you only need to take the following steps.  
- Model initialization  
- Calling the `predict` interface  
- Visualization validation (optional)

|Image Classification|Object Detection|OCR Text Recognition|Portrait Segmentation|Face Detection|  
|:---:|:---:|:---:|:---:|:---:|
|![classify](https://user-images.githubusercontent.com/31974251/203261658-600bcb09-282b-4cd3-a2f2-2c733a223b03.gif)|![detection](https://user-images.githubusercontent.com/31974251/203261763-a7513df7-e0ab-42e5-ad50-79ed7e8c8cd2.gif)|![ocr](https://user-images.githubusercontent.com/31974251/203261817-92cc4fcd-463e-4052-910c-040d586ff4e7.gif)|![seg](https://user-images.githubusercontent.com/31974251/203267867-7c51b695-65e6-402e-9826-5d6d5864da87.gif)|![face](https://user-images.githubusercontent.com/31974251/203261714-c74631dd-ec5b-4738-81a3-8dfc496f7547.gif)|

## Content

- [Download and Configure SDK](#SDK)
- [Image Classification API](#Classification)  
- [Object Detection API](#Detection)  
- [Semantic Segmentation API](#Segmentation)  
- [OCR Text Recognition API ](#OCR)  
- [Face Detection API](#FaceDetection)  
- [Identification Result Description](#VisionResults)
- [Runtime Option Description](#RuntimeOption)  
- [Visualization Interface ](#Visualize)
- [Examples of How to Use Models](#Demo)
- [How to Use the App Sample Project](#App)  

## Download and Configure SDK
<div id="SDK"></div>  

### Download FastDeploy Android SDK  
The release version is as follows (Java SDK currently supports Android only):

| Platform | File | Description |
| :--- | :--- | :---- |
| Android Java SDK | [fastdeploy-android-sdk-0.0.0.aar](https://bj.bcebos.com/fastdeploy/dev/android/fastdeploy-android-sdk-0.0.0.aar) | NDK 20 compiles, minSdkVersion 15,Object SdkVersion 28 |

For more information for pre-compile library, please refer to: [download_prebuilt_libraries.md](../../docs/cn/build_and_install/download_prebuilt_libraries.md).

## Configure FastDeploy Android SDK  

First, please copy fastdeploy-android-sdk-xxx.aar to the libs directory of your Android project, where `xxx` indicates the version number of the SDK you download.
```shell
├── build.gradle
├── libs
│   └── fastdeploy-android-sdk-xxx.aar
├── proguard-rules.pro
└── src
```

Then, please add FastDeploy SDK to build.gradble in your Android project.
```java  
dependencies {
    implementation fileTree(include: ['*.aar'], dir: 'libs')
    implementation 'com.android.support:appcompat-v7:28.0.0'
    // ...
}
```

## Image Classification API

<div id="Classification"></div>  

###  PaddleClasModel Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PaddleClasModel initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - configFile: String, preprocessing configuration file of model inference, e.g. infer_cfg.yml.  
  - labelFile: String, optional, path to the label file, for visualization, e.g. imagenet1k_label_list.txt, in which each line contains a label.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  


```java
// Constructor w/o label file
public PaddleClasModel(); // An empty constructor, which can be initialised by calling init function later.
public PaddleClasModel(String modelFile, String paramsFile, String configFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PaddleClasModel(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// Call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public ClassifyResult predict(Bitmap ARGB8888Bitmap)；
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public ClassifyResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public ClassifyResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // Only rendering images without saving.
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources.
public boolean initialized(); // Check if initialization is successful.
```

## Object Detection API

<div id="Detection"></div>  

### PicoDet Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PicoDet initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - configFile: String, preprocessing configuration file of model inference, e.g. infer_cfg.yml.  
  - labelFile: String, optional, path to the label file, for visualization, e.g. coco_label_list.txt, in which each line contains a label.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  
  
```java
// Constructor w/o label file.
public PicoDet(); // An empty constructor, which can be initialised by calling init function later.
public PicoDet(String modelFile, String paramsFile, String configFile);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile);
public PicoDet(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public PicoDet(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
// Call init manually w/o label file.
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
public boolean init(String modelFile, String paramsFile, String configFile, String labelFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public DetectionResult predict(Bitmap ARGB8888Bitmap)；
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public DetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float scoreThreshold);
public DetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float scoreThreshold); // Only rendering images without saving.
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources. 
public boolean initialized(); // Check if initialization is successful.
```

## OCR Text Recognition API  

<div id="OCR"></div>  

### PP-OCRv2 & PP-OCRv3 Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PP-OCR initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - labelFile: String, optional, path to the label file, for visualization, e.g. ppocr_keys_v1.txt, in which each line contains a label.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  
Unlike other models, PP-OCRv2 and PP-OCRv3 contain base models such as DBDetector, Classifier and Recognizer, and pipeline types such as PPOCRv2 and PPOCRv3.
```java
// Constructor w/o label file
public DBDetector(String modelFile, String paramsFile);
public DBDetector(String modelFile, String paramsFile, RuntimeOption option);
public Classifier(String modelFile, String paramsFile);
public Classifier(String modelFile, String paramsFile, RuntimeOption option);
public Recognizer(String modelFile, String paramsFile, String labelPath);
public Recognizer(String modelFile, String paramsFile,  String labelPath, RuntimeOption option);
public PPOCRv2();  // An empty constructor, which can be initialised by calling init function later.
// Constructor w/o classifier
public PPOCRv2(DBDetector detModel, Recognizer recModel);
public PPOCRv2(DBDetector detModel, Classifier clsModel, Recognizer recModel);
public PPOCRv3();  // An empty constructor, which can be initialised by calling init function later.
// Constructor w/o classifier
public PPOCRv3(DBDetector detModel, Recognizer recModel);
public PPOCRv3(DBDetector detModel, Classifier clsModel, Recognizer recModel);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public OCRResult predict(Bitmap ARGB8888Bitmap)；
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public OCRResult predict(Bitmap ARGB8888Bitmap, String savedImagePath);
public OCRResult predict(Bitmap ARGB8888Bitmap, boolean rendering); // Only rendering images without saving.
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources.  
public boolean initialized(); // Check if initialization is successful.
```

## Semantic Segmentation API  

<div id="Segmentation"></div>  

### PaddleSegModel Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PaddleSegModel initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - configFile: String, preprocessing configuration file of model inference, e.g. infer_cfg.yml.  
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  

```java
// Constructor w/o label file
public PaddleSegModel(); // An empty constructor, which can be initialised by calling init function later.
public PaddleSegModel(String modelFile, String paramsFile, String configFile);
public PaddleSegModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
// Call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public SegmentationResult predict(Bitmap ARGB8888Bitmap)；
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public SegmentationResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float weight);
public SegmentationResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float weight); // Only rendering images without saving.
// Modify result, but not return it. Concerning performance, you can use the following interface with CxxBuffer in SegmentationResult.
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result)；
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, String savedImagePath, float weight);
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, boolean rendering, float weight);
```
- Set vertical or horizontal mode: For PP-HumanSeg series model, you should call this method to set the vertical mode to true.
```java  
public void setVerticalScreenFlag(boolean flag);
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources.
public boolean initialized(); // Check if initialization is successful.
```

## Face Detection API

<div id="FaceDetection"></div>  

### SCRFD Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PaddleSegModel initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  

```java
// Constructor w/o label file.
public SCRFD(); // An empty constructor, which can be initialised by calling init function later.
public SCRFD(String modelFile, String paramsFile);
public SCRFD(String modelFile, String paramsFile, RuntimeOption option);
// Call init manually w/o label file.
public boolean init(String modelFile, String paramsFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap)；
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, float confThreshold, float nmsIouThreshold)； // Set confidence thresholds and NMS thresholds.
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float confThreshold, float nmsIouThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float confThreshold, float nmsIouThreshold); // Only rendering images without saving.
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources. 
public boolean initialized(); // Check if initialization is successful.
```

### YOLOv5Face Java API Introduction
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PaddleSegModel initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.  

```java
// Constructor w/o label file.
public YOLOv5Face(); // An empty constructor, which can be initialised by calling init function later.
public YOLOv5Face(String modelFile, String paramsFile);
public YOLOv5Face(String modelFile, String paramsFile, RuntimeOption option);
// Call init manually w/o label file.
public boolean init(String modelFile, String paramsFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap)；
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, float confThreshold, float nmsIouThreshold)； // Set confidence thresholds and NMS thresholds.
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float confThreshold, float nmsIouThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float confThreshold, float nmsIouThreshold); // Only rendering images without saving.
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources.  
public boolean initialized(); // Check if initialization is successful.
```

## Identification Result Description

<div id="VisionResults"></div>  

- Image classification result description
```java
public class ClassifyResult {
  public float[] mScores;  // [n]   Scores of every class(probability).
  public int[] mLabelIds;  // [n]   Class ID, specific class type.
  public boolean initialized(); // To test whether the result is valid.
}
```  
Other reference: C++/Python corresponding ClassifyResult description: [api/vision_results/classification_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result.md)

- Object detection result description
```java
public class DetectionResult {
  public float[][] mBoxes; // [n,4] Detecting box (x1,y1,x2,y2).
  public float[] mScores;  // [n]   Score (confidence level, probability value) for each detecting box.
  public int[] mLabelIds;  // [n]   Class ID.
  public boolean initialized(); // To test whether the result is valid.
}
```  
Other reference: C++/Python corresponding DetectionResult description:  [api/vision_results/detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/detection_result.md)

- OCR text recognition result description
```java
public class OCRResult {
  public int[][] mBoxes;  // [n,8] indicates the coordinates of all Object boxes detected in a single image. Each box is 8 int values representing the 4 coordinate points of the box, in the order of lower left, lower right, upper right, upper left.
  public String[] mText;  // [n] indicates the content recognized in multiple text boxes. 
  public float[] mRecScores;  // [n] indicates the confidence level of the text recognized in the text box.
  public float[] mClsScores;  // [n] indicates the confidence level of the classification result of the text.
  public int[] mClsLabels;  // [n] indicates the direction classification category of the text box.
  public boolean initialized(); // To test whether the result is valid.
}
```  
Other reference: C++/Python corresponding OCRResult description: [api/vision_results/ocr_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/ocr_result.md)

- Semantic segmentation result description
```java
public class SegmentationResult {
  public int[] mLabelMap;  //  The predicted label map, each pixel position corresponds to a label HxW.
  public float[] mScoreMap; // The predicted score map, each pixel position corresponds to a score HxW.
  public long[] mShape; // The real shape(H,W) of label map.
  public boolean mContainScoreMap = false; // Whether score map is included.
  // You can choose to use CxxBuffer directly instead of copying it to JAVA layer through JNI.
  // This method can improve performance to some extent.
  public void setCxxBufferFlag(boolean flag); // Set whether the mode is CxxBuffer.
  public boolean releaseCxxBuffer(); // Release CxxBuffer manually!!!
  public boolean initialized(); // Check if the result is valid.
}  
```
Other reference: C++/Python corresponding SegmentationResult description: [api/vision_results/segmentation_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/segmentation_result.md)

- Face detection result description  
```java
public class FaceDetectionResult {
  public float[][] mBoxes; // [n,4] detection box (x1,y1,x2,y2)
  public float[] mScores;  // [n]  scores(confidence level, probability value) of every detection box 
  public float[][] mLandmarks; // [nx?,2] Each detected face corresponding keypoint
  int mLandmarksPerFace = 0;  // Each face corresponding keypoints number
  public boolean initialized(); // Check if the result is valid.
}  
```
Other reference：C++/Python corresponding FaceDetectionResult description:  [api/vision_results/face_detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/face_detection_result.md)

## Runtime Option Description

<div id="RuntimeOption"></div>  

- RuntimeOption setting description 
```java
public class RuntimeOption {
  public void enableLiteFp16(); // Enable fp16 precision inference
  public void disableLiteFP16(); // Disable fp16 precision inference
  public void enableLiteInt8(); // Enable int8 precision inference, for quantized models
  public void disableLiteInt8(); // Disable int8 precision inference
  public void setCpuThreadNum(int threadNum); // Set number of threads.
  public void setLitePowerMode(LitePowerMode mode);  // Set power mode.
  public void setLitePowerMode(String modeStr);  // Set power mode by string.
}
```

## Visualization Interface 

<div id="Visualize"></div>  

FastDeploy Android SDK also provides visual interfaces that can be used to quickly validate the inference results. The following interfaces all render the result in the input Bitmap.

```java  
public class Visualize {
  // Default parameter interface.
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result);
  public static boolean visFaceDetection(Bitmap ARGB8888Bitmap, FaceDetectionResult result);
  public static boolean visOcr(Bitmap ARGB8888Bitmap, OCRResult result);
  public static boolean visSegmentation(Bitmap ARGB8888Bitmap, SegmentationResult result);
  // Visual interface with configurable parameters. 
  // visDetection: You can configure the threshold value (draw the boxes higher than the threshold), box line size, font size, labels, etc.
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, float scoreThreshold);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, float scoreThreshold, int lineSize, float fontSize);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, String[] labels);
  public static boolean visDetection(Bitmap ARGB8888Bitmap, DetectionResult result, String[] labels, float scoreThreshold, int lineSize, float fontSize);
  // visClassification: You can configure the threshold value (draw the boxes higher than the threshold), font size, labels, etc.
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, float scoreThreshold,float fontSize);
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, String[] labels);
  public static boolean visClassification(Bitmap ARGB8888Bitmap, ClassifyResult result, String[] labels, float scoreThreshold,float fontSize);
  // visSegmentation: Background weight.
  public static boolean visSegmentation(Bitmap ARGB8888Bitmap, SegmentationResult result, float weight);
  // visFaceDetection: String size, font size, etc.
  public static boolean visFaceDetection(Bitmap ARGB8888Bitmap, FaceDetectionResult result, int lineSize, float fontSize);
}
```  
The corresponding visualization types:  
```java
import com.baidu.paddle.fastdeploy.vision.Visualize;
```

## Examples of How to Use Models

<div id="Demo"></div>  

- Example 1: Using constructor function and default RuntimeOption.
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;

// Initialize model.
PicoDet model = new PicoDet("picodet_s_320_coco_lcnet/model.pdmodel",
                            "picodet_s_320_coco_lcnet/model.pdiparams",
                            "picodet_s_320_coco_lcnet/infer_cfg.yml");

// Model inference.
DetectionResult result = model.predict(ARGB8888ImageBitmap);  

// Release model resources.
model.release();
```  

- Example 2: Manually call init function at appropriate program nodes, and customize RuntimeOption.
```java  
// import id.
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.DetectionResult;
import com.baidu.paddle.fastdeploy.vision.detection.PicoDet;
// Create a new empty model.
PicoDet model = new PicoDet();  
// Model path.
String modelFile = "picodet_s_320_coco_lcnet/model.pdmodel";
String paramFile = "picodet_s_320_coco_lcnet/model.pdiparams";
String configFile = "picodet_s_320_coco_lcnet/infer_cfg.yml";
// Set RuntimeOption.
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();
// Initiaze with init function.
model.init(modelFile, paramFile, configFile, option);
// Reading Bitmap, model prediction, resource release id.
```

## How to Use the App Sample Project
<div id="App"></div>  

FastDeploy provides some sample projects in the java/android/app directory. Since the java/android directory also contains JNI projects, users who want to use the sample projects also need to configure the NDK. If you only want to use the Java API and don't want to configure the NDK, you can jump to the detailed case links below.

- [App sample project of image classification](../../examples/vision/classification/paddleclas/android)  
- [App sample project of Object detection](../../examples/vision/detection/paddledetection/android)  
- [App sample project of OCR text detection](../../examples/vision/ocr/PP-OCRv2/android)  
- [App sample project of portrait segmentation](../../examples/vision/segmentation/paddleseg/android)  
- [App sample project of face detection](../../examples/vision/facedet/scrfd/android)  

### Prepare for Environment

1. Install Android Studio tools in your local environment, please refer to [Android Stuido official website](https://developer.android.com/studio) for detailed installation method.
2. Get an Android phone and turn on USB debugging mode. How to turn on: ` Phone Settings -> Find Developer Options -> Turn on Developer Options and USB Debug Mode`.


**Notes**：If your Android Studio is not configured with an NDK, please configure the it according to [Installing and Configuring NDK and CMake](https://developer.android.com/studio/projects/install-ndk) in the Android Studio User Guide. You can either choose the latest NDK version or use the same version as the FastDeploy Android prediction library.

### Configuration Steps

1. The App sample project is located in directory `fastdeploy/java/android/app`.
2. Open `fastdeploy/java/android` project by Android Studio, please note that the directory is `java/android`.
3. Connect your phone to your computer, turn on USB debugging and file transfer mode, and connect your own mobile device on Android Studio (your phone needs to be enabled to allow software installation from USB).

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **Notes:**
>> If you encounter an NDK configuration error during importing, compiling or running the program, please open ` File > Project Structure > SDK Location` and change `Andriod NDK location` to your locally configured NDK path. The default NDK version in this project is 20.
>> If you downloaded the NDK through SDK Tools in Andriod Studio (see "Prepare for Environment" in this section), you can select the default path by clicking the drop-down box.
>> There is another way to configure the NDK: you can do it manually in the file `java/android/local.properties`, as shown above.
>> If the above steps still can't solve the configuration error, please try to update Android Gradle plugin version according to section [Updating Android Gradle plugin](https://developer.android.com/studio/releases/gradle-plugin?hl=zh-cn#updating-plugin) in official Andriod Studio documentation.


4. Click the Run button to automatically compile the APP and install it to your phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files, internet connection required.)
The success interface is as follows. Figure 1: Install APP on phone; Figure 2: The opening interface, it will automatically recognize the objects in the picture and mark them; Figure 3: APP setting options, click setting in the upper right corner, and you can set different options.

  | APP icon | APP effect | APP setting options
  | ---     | --- | --- |
  | ![app_pic](https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg)   | ![app_res](https://user-images.githubusercontent.com/31974251/197169609-bb214af3-d6e7-4433-bb96-1225cddd441c.jpg) |  ![app_setup](https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg) |  

### Switch Between Different Scenarios  
App sample project only needs to switch between different Activity in AndroidManifest.xml to compile App in different scenarios.

<p align="center">
<img width="788" alt="image" src="https://user-images.githubusercontent.com/31974251/203258255-b422d3e2-6004-465f-86b6-9fa61a27c6c2.png">
</p>  

- Image classification scenario
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
- Object detection scenario
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
- OCR text detection scenario 
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
- Portrait segmentation scenario
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
- Face detection scenario
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
