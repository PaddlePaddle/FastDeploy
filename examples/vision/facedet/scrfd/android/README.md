English | [简体中文](README_CN.md)
# Target Detection SCRFD Android Demo Tutorial

Real-time target detection on Android. This Demo is simple to use for everyone. For example, you can run your own trained model in the Demo.

## Prepare the Environment

1. Install Android Studio in your local environment. Refer to [Android Stuido Official Website](https://developer.android.com/studio) for detailed tutorial.
2. Prepare an Android phone and turn on the USB debug mode: `Settings -> Find developer options -> Open developer options and USB debug mode`

## Deployment steps

1. The target detection SCRFD Demo is located in the `fastdeploy/examples/vision/facedet/scrfd/android` directory
2. Use Android Studio to open the scrfd/android project
3. Connect the phone to the computer, turn on USB debug mode and file transfer mode, and connect your phone to Android Studio (allow the phone to install software from USB)

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **Attention：**
>> If you encounter an NDK configuration error during import, compilation or running, open ` File > Project Structure > SDK Location`  and change the path of SDK configured by the `Andriod SDK location`.

4. Click the Run button to automatically compile the APP and install it to the phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files. Internet is required). 
The final effect is as follows. Figure 1: Install the APP on the phone; Figure 2: The effect when opening the APP. It will automatically recognize and mark the objects in the image; Figure 3: APP setting option. Click setting in the upper right corner and modify your options.

 | APP Icon | APP Effect | APP Settings
  | ---     | --- | --- |
  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg">  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203261714-c74631dd-ec5b-4738-81a3-8dfc496f7547.gif"> | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg"> |  

## SCRFD Java API Description
- Model initialized API: The initialized API contains two ways: Firstly, initialize directly through the constructor. Secondly, initialize at the appropriate program node by calling the init function. PicoDet initialization parameters are as follows: 
  - modelFile: String. Model file path in paddle format, such as model.pdmodel
  - paramFile: String. Parameter file path in paddle format, such as model.pdiparams 
  - option: RuntimeOption. Optional parameter for model initialization. Default runtime options if not passing the parameter. 

```java
// Constructor: constructor w/o label file
public SCRFD(); // An empty constructor. It can be initialized later by calling init function.
public SCRFD(String modelFile, String paramsFile);
public SCRFD(String modelFile, String paramsFile, RuntimeOption option);
// Call init manually for initialization: call init manually w/o label file
public boolean init(String modelFile, String paramsFile, RuntimeOption option);
```  
- Model Prediction API: The Model Prediction API contains an API for direct prediction and an API for visualization. In direct prediction, we do not save the image and render the result on Bitmap. Instead, we merely predict the inference result. For prediction and visualization, the results are both predicted and visualized, the visualized images are saved to the specified path, and the visualized results are rendered in Bitmap (Now Bitmap in ARGB8888 format is supported). Afterward, the Bitmap can be displayed on the camera.
```java
// Direct prediction: No image saving and no result rendering to Bitmap 
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap)；
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, float confThreshold, float nmsIouThreshold)； // Set confidence threshold and NMS threshold
// Prediction and visualization: Predict and visualize the results, save the visualized image to the specified path, and render the visualized results on Bitmap 
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float confThreshold, float nmsIouThreshold);
public FaceDetectionResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float confThreshold, float nmsIouThreshold); // Render without saving images
```
- Model resource release API: Call release() API to release model resources. Return true for successful release and false for failure; call initialized() to determine whether the model was initialized successfully, with true indicating successful initialization and false indicating failure. 
```java
public boolean release(); // Release native resources 
public boolean initialized(); // Check if initialization is successful
```

- RuntimeOption settings
```java  
public void enableLiteFp16(); //  Enable fp16 accuracy inference
public void disableLiteFP16(); // Disable fp16 accuracy inference
public void setCpuThreadNum(int threadNum); // Set thread numbers
public void setLitePowerMode(LitePowerMode mode);  // Set power mode
public void setLitePowerMode(String modeStr);  // Set power mode through character string
```

- Model SegmentationResult
```java
public class FaceDetectionResult {
  public float[][] mBoxes; // [n,4] Detection box (x1,y1,x2,y2)
  public float[] mScores;  // [n] Score of each detection box (confidence, probability)
  public float[][] mLandmarks; // [nx?,2] Each detected face keypoints
  int mLandmarksPerFace = 0;  // Number of detected face keypoints
  public boolean initialized(); // Whether the result is valid
}  
```  
Refer to [api/vision_results/face_detection_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/face_detection_result.md) for C++/Python DetectionResult


- Model Calling Example 1: Using Constructor and the default RuntimeOption
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.FaceDetectionResult;
import com.baidu.paddle.fastdeploy.vision.facedet.SCRFD;

// Initialize the model
SCRFD model = new SCRFD(
  "scrfd_500m_bnkps_shape320x320_pd/model.pdmodel",
  "scrfd_500m_bnkps_shape320x320_pd/model.pdiparams");

// Read the image: The following is merely the pseudo code to read Bitmap
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// Return FaceDetectionResult after direct prediction
FaceDetectionResult result = model.predict(ARGB8888ImageBitmap);

// Release model resources  
model.release();
```  

- Model calling example 2: Manually call init at the appropriate program node and self-define RuntimeOption
```java  
// import is as the above...
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.FaceDetectionResult;
import com.baidu.paddle.fastdeploy.vision.facedet.SCRFD;
// Create an empty model
SCRFD model = new SCRFD();  
// Model path
String modelFile = "scrfd_500m_bnkps_shape320x320_pd/model.pdmodel";
String paramFile = "scrfd_500m_bnkps_shape320x320_pd/model.pdiparams";
// Specify RuntimeOption
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();  
// Use init function for initialization  
model.init(modelFile, paramFile, option);
// Bitmap reading, model prediction, and resource release are as above ...
```
Refer to [FaceDetMainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/facedet/FaceDetMainActivity.java) for more details

## Replace FastDeploy SDK and Models
It’s simple to replace the FastDeploy prediction library and models. The prediction library is located at `app/libs/fastdeploy-android-sdk-xxx.aar`, where `xxx` represents the version of your prediction library. The models are located at `app/src/main/assets/models/scrfd_500m_bnkps_shape320x320_pd`。  
-  Replace the FastDeploy Android SDK: Download or compile the latest FastDeploy Android SDK, unzip and place it in the `app/libs` directory; For detailed configuration, refer to  
     - [FastDeploy Java SDK in Android](../../../../../java/android/)

- Steps to replace SCRFD models: 
  - Other SCRFD models exported by X2Paddle, refer to [SCRFD document](../README.md) and [X2Paddle](https://github.com/PaddlePaddle/X2Paddle)  
  - Put your SCRFD model in `app/src/main/assets/models` directory;  
  - Modify the default value of the model path in `app/src/main/res/values/strings.xml`. For example, 
```xml
<!-- Change this path to your model, such as models/scrfd_500m_bnkps_shape320x320_pd -->
<string name="FACEDET_MODEL_DIR_DEFAULT">models/scrfd_500m_bnkps_shape320x320_pd</string>  
```  

## More Reference Documents
For more FastDeploy Java API documentes and how to access FastDeploy C++ API via JNI, refer to: 
- [FastDeploy Java SDK in Android](../../../../../java/android/)
- [FastDeploy C++ SDK in Android](../../../../../docs/en/faq/use_cpp_sdk_on_android.md)  
