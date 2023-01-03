English | [简体中文](README.md)
# OCR Text Recognition Android Demo Tutorial 

Real-time OCR text recognition on Android. This demo is easy to use for everyone. For example, you can run your own trained model in the demo.

## Prepare the Environment

1. Install Android Studio in your local environment. Refer to [Android Studio Official Website](https://developer.android.com/studio) for detailed tutorial.
2. Prepare an Android phone and turn on the USB debug mode. Opening: `Settings -> Find developer options ->  Open developer options and USB debug mode`

## Deployment steps

1. The OCR text recognition Demo is located in the `fastdeploy/examples/vision/ocr/PP-OCRv3/android`
2. Open PP-OCRv2/android project with Android Studio
3. Connect the phone to the computer, turn on USB debug mode and file transfer mode, and connect your phone to Android Studio (allow the phone to install software from USB)

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **Attention：**
>> If you encounter an NDK configuration error during import, compilation or running, open ` File > Project Structure > SDK Location` and change the path of SDK configured by the `Andriod SDK location` 

4. Click the Run button to automatically compile the APP and install it to the phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files. Internet is required). 
The final effect is as follows. Figure 1: Install the APP on the phone; Figure 2: The effect after opening the APP. It will automatically recognize and mark the objects in the image; Figure 3: APP setting option. Click setting in the upper right corner and modify your options.

| APP Icon | APP Effect | APP Settings
  | ---     | --- | --- |
| ![app_pic](https://user-images.githubusercontent.com/14995488/203484427-83de2316-fd60-4baf-93b6-3755f9b5559d.jpg)   | ![app_res](https://user-images.githubusercontent.com/14995488/203495616-af42a5b7-d3bc-4fce-8d5e-2ed88454f618.jpg) |  ![app_setup](https://user-images.githubusercontent.com/14995488/203484436-57fdd041-7dcc-4e0e-b6cb-43e5ac1e729b.jpg) |  

### PP-OCRv3 Java API Description

- Model initialized API: The initialized API contains two ways: Firstly, initialize directly through the constructor. Secondly, initialize at the appropriate program node by calling the init function. PP-OCR initialization parameters are as follows:
  - modelFile: String. Model file path in paddle format, such as model.pdmodel
  - paramFile: String. Parameter file path in paddle format, such as model.pdiparams
  - labelFile: String. This optional parameter indicates the path of the label file and is used for visualization. such as ppocr_keys_v1.txt, each line containing one label
  - option: RuntimeOption. Optional parameter for model initialization. Default runtime options if the parameter is not passed. Different from other models, PP-OCRv3 contains base models such as DBDetector, Classifier, Recognizer and the pipeline type.
```java
// Constructor: constructor w/o label file
public DBDetector(String modelFile, String paramsFile);
public DBDetector(String modelFile, String paramsFile, RuntimeOption option);
public Classifier(String modelFile, String paramsFile);
public Classifier(String modelFile, String paramsFile, RuntimeOption option);
public Recognizer(String modelFile, String paramsFile, String labelPath);
public Recognizer(String modelFile, String paramsFile,  String labelPath, RuntimeOption option);
public PPOCRv3();  // An empty constructor, which can be initialized by calling init
// Constructor w/o classifier
public PPOCRv3(DBDetector detModel, Recognizer recModel);
public PPOCRv3(DBDetector detModel, Classifier clsModel, Recognizer recModel);
```  
- Model Prediction API: The Model Prediction API contains an API for direct prediction and an API for visualization. In direct prediction, we do not save the image and render the result on Bitmap. Instead, we merely predict the inference result. For prediction and visualization, the results are both predicted and visualized, the visualized images are saved to the specified path, and the visualized results are rendered in Bitmap (Now Bitmap in ARGB8888 format is supported). Afterward, the Bitmap can be displayed on the camera.
```java
// Direct prediction: No image saving and no result rendering to Bitmap 
public OCRResult predict(Bitmap ARGB8888Bitmap)；
// Prediction and visualization: Predict and visualize the results, save the visualized image to the specified path, and render the visualized results on Bitmap 
public OCRResult predict(Bitmap ARGB8888Bitmap, String savedImagePath);
public OCRResult predict(Bitmap ARGB8888Bitmap, boolean rendering); // Render without saving images
```
- Model resource release API: Call release() API to release model resources. Return true for successful release and false for failure; call initialized() to determine whether the model was initialized successfully, with true indicating successful initialization and false indicating failure. 
```java
public boolean release(); // Realise native resources 
public boolean initialized(); // Check if initialization was successful
```

- RuntimeOption settings

```java  
public void enableLiteFp16(); // Enable fp16 accuracy inference
public void disableLiteFP16(); // Disable fp16 accuracy inference
public void enableLiteInt8(); // Enable int8 accuracy inference for quantification models
public void disableLiteInt8(); // Disable int8 accuracy inference
public void setCpuThreadNum(int threadNum); // Set thread numbers
public void setLitePowerMode(LitePowerMode mode);  // Set power mode
public void setLitePowerMode(String modeStr);  // Set power mode through character string
```

- Model OCRResult
```java
public class OCRResult {
  public int[][] mBoxes; // The coordinates of all target boxes in a single image. 8 int values represent the 4 coordinate points of the box in the order of bottom left, bottom right, top right and top left
  public String[] mText; // Recognized text in multiple text boxes
  public float[] mRecScores; // Confidence of the recognized text in the box
  public float[] mClsScores; // Confidence of the classification result of the text box
  public int[] mClsLabels; // Directional classification of the text box
  public boolean mInitialized = false; // Whether the result is valid or not
}  
```
Refer to [api/vision_results/ocr_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/ocr_result.md) for C++/Python OCRResult


- Model Calling Example 1: Using Constructor
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.OCRResult;
import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;

// Model path
String detModelFile = "ch_PP-OCRv3_det_infer/inference.pdmodel";
String detParamsFile = "ch_PP-OCRv3_det_infer/inference.pdiparams";
String clsModelFile = "ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
String clsParamsFile = "ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams";
String recModelFile = "ch_PP-OCRv3_rec_infer/inference.pdmodel";
String recParamsFile = "ch_PP-OCRv3_rec_infer/inference.pdiparams";
String recLabelFilePath = "labels/ppocr_keys_v1.txt";
// Set the RuntimeOption
RuntimeOption detOption = new RuntimeOption();
RuntimeOption clsOption = new RuntimeOption();
RuntimeOption recOption = new RuntimeOption();
detOption.setCpuThreadNum(2);
clsOption.setCpuThreadNum(2);
recOption.setCpuThreadNum(2);
detOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
clsOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
recOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
detOption.enableLiteFp16();  
clsOption.enableLiteFp16();  
recOption.enableLiteFp16();  
// Initialize the model
DBDetector detModel = new DBDetector(detModelFile, detParamsFile, detOption);
Classifier clsModel = new Classifier(clsModelFile, clsParamsFile, clsOption);
Recognizer recModel = new Recognizer(recModelFile, recParamsFile, recLabelFilePath, recOption);
PPOCRv3 model = new PPOCRv3(detModel，clsModel，recModel);

// Read the image: The following is merely the pseudo code to read the Bitmap
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// Model Inference
OCRResult result = model.predict(ARGB8888ImageBitmap);  

// Release model resources  
model.release();
```  

- Model calling example 2: Manually call init at the appropriate program node
```java  
// import is as above...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.OCRResult;
import com.baidu.paddle.fastdeploy.vision.ocr.Classifier;
import com.baidu.paddle.fastdeploy.vision.ocr.DBDetector;
import com.baidu.paddle.fastdeploy.vision.ocr.Recognizer;
// Create an empty model
PPOCRv3 model = new PPOCRv3();  
// Model path
String detModelFile = "ch_PP-OCRv3_det_infer/inference.pdmodel";
String detParamsFile = "ch_PP-OCRv3_det_infer/inference.pdiparams";
String clsModelFile = "ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
String clsParamsFile = "ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams";
String recModelFile = "ch_PP-OCRv3_rec_infer/inference.pdmodel";
String recParamsFile = "ch_PP-OCRv3_rec_infer/inference.pdiparams";
String recLabelFilePath = "labels/ppocr_keys_v1.txt";
// Set the RuntimeOption
RuntimeOption detOption = new RuntimeOption();
RuntimeOption clsOption = new RuntimeOption();
RuntimeOption recOption = new RuntimeOption();
detOption.setCpuThreadNum(2);
clsOption.setCpuThreadNum(2);
recOption.setCpuThreadNum(2);
detOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
clsOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
recOption.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
detOption.enableLiteFp16();  
clsOption.enableLiteFp16();  
recOption.enableLiteFp16();  
// Use init function for initialization
DBDetector detModel = new DBDetector(detModelFile, detParamsFile, detOption);
Classifier clsModel = new Classifier(clsModelFile, clsParamsFile, clsOption);
Recognizer recModel = new Recognizer(recModelFile, recParamsFile, recLabelFilePath, recOption);
model.init(detModel, clsModel, recModel);
// Bitmap reading, model prediction, and resource release are as above ...
```
Refer to [OcrMainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/ocr/OcrMainActivity.java) for more details

## Replace FastDeploy SDK and Models
It’s simple to replace the FastDeploy prediction library and models. The prediction library is located at `app/libs/fastdeploy-android-sdk-xxx.aar`, where `xxx`  represents the version of your prediction library. The models are located at `app/src/main/assets/models`.
- Replace the FastDeploy Android SDK: Download or compile the latest FastDeploy Android SDK, unzip and place it in the `app/libs`; For detailed configuration, refer to 
  - [FastDeploy Java SDK in Android](../../../../../java/android/)

- Steps to replace OCR models:
  - Put your OCR model in `app/src/main/assets/models`;
  - Modify the default value of the model path in `app/src/main/res/values/strings.xml`. For example,
```xml
<!-- Change this path to your model -->
<string name="OCR_MODEL_DIR_DEFAULT">models</string>  
<string name="OCR_LABEL_PATH_DEFAULT">labels/ppocr_keys_v1.txt</string>
```  
## Use quantification models
If you're using quantification models, set Int8 accuracy inference using the interface enableLiteInt8() of RuntimeOption.
```java
String detModelFile = "ch_ppocrv3_plate_det_quant/inference.pdmodel";
String detParamsFile = "ch_ppocrv3_plate_det_quant/inference.pdiparams";
String recModelFile = "ch_ppocrv3_plate_rec_distillation_quant/inference.pdmodel";
String recParamsFile = "ch_ppocrv3_plate_rec_distillation_quant/inference.pdiparams";
String recLabelFilePath = "ppocr_keys_v1.txt"; // ppocr_keys_v1.txt
RuntimeOption detOption = new RuntimeOption();
RuntimeOption recOption = new RuntimeOption();
// Use Int8 accuracy for inference
detOption.enableLiteInt8();
recOption.enableLiteInt8();
// Initialize PP-OCRv3 Pipeline
PPOCRv3 predictor = new PPOCRv3();
DBDetector detModel = new DBDetector(detModelFile, detParamsFile, detOption);
Recognizer recModel = new Recognizer(recModelFile, recParamsFile, recLabelFilePath, recOption);
predictor.init(detModel, recModel);
```
Refer to [OcrMainActivity.java](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/ocr/OcrMainActivity.java) for use-pattern in APP.

## More Reference Documents
For more FastDeploy Java API documentes and how to access FastDeploy C++ API via JNI, refer to:
- [FastDeploy Java SDK in Android](../../../../../java/android/)
- [FastDeploy C++ SDK in Android](../../../../../docs/cn/faq/use_cpp_sdk_on_android.md)  
