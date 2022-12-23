English | [简体中文](README.md)

# FastDeploy Android AAR Package  
Currently FastDeploy Android SDK supports image classification, target detection, OCR text recognition, semantic segmentation and face detection. More AI tasks will be added in the future. The following is the API documents for each task. To use the models integrated in FastDeploy on Android, you only need to take the following steps.  
- Model initialization  
- Calling the `predict` interface  
- Visualization validation (optional)

|Image Classification|Target Detection|OCR Text Recognition|Portrait Segmentation|Face Detection|  
|:---:|:---:|:---:|:---:|:---:|
|![classify](https://user-images.githubusercontent.com/31974251/203261658-600bcb09-282b-4cd3-a2f2-2c733a223b03.gif)|![detection](https://user-images.githubusercontent.com/31974251/203261763-a7513df7-e0ab-42e5-ad50-79ed7e8c8cd2.gif)|![ocr](https://user-images.githubusercontent.com/31974251/203261817-92cc4fcd-463e-4052-910c-040d586ff4e7.gif)|![seg](https://user-images.githubusercontent.com/31974251/203267867-7c51b695-65e6-402e-9826-5d6d5864da87.gif)|![face](https://user-images.githubusercontent.com/31974251/203261714-c74631dd-ec5b-4738-81a3-8dfc496f7547.gif)|

## Content













## Download and Configure SDK
<div id="SDK"></div>  

### Download FastDeploy Android SDK  
The release version is as follows (Java SDK currently supports Android only, and current version is 1.0.0):

| Platform | File | Description |
| :--- | :--- | :---- |
| Android Java SDK | [fastdeploy-android-sdk-1.0.0.aar](https://bj.bcebos.com/fastdeploy/release/android/fastdeploy-android-sdk-1.0.0.aar) | NDK 20 compiles, minSdkVersion 15,targetSdkVersion 28 |

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

### Description of PaddleClasModel Java API 
- Model initialization API: Model initialization API consists of two ways, one is to initialize directly through the constructor, and the other is to initialize at the appropriate program node by calling the init function. paddleClasModel initialization parameters are described as follows  
  - modelFile: String, the path to the model file in paddle format, such as model.pdmodel
  - paramFile: String, the path of the parameter file in paddle format, such as model.pdiparams  
  - configFile: String, the preprocessing configuration file of the model inference, such as infer_cfg.yml  
  - labelFile: String, an optional parameter indicating the path to the label file for visualization, e.g. imagenet1k_label_list.txt, each line contains a label  
  - option: RuntimeOption, optional parameter, model initialization option. if this parameter is not passed the default runtime option will be used.  