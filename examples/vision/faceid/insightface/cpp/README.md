English | [简体中文](README_CN.md)
# InsightFace C++ Deployment Example
This direct ory provides examples that `infer_xxx.cc` fast finishes the deployment of InsighFace, including ArcFace\CosFace\VPL\Partial_FC on CPU/GPU and GPU accelerated by TensorRT. 
Taking ArcFace as an example, we demonstrate how `infer_arcface.cc` fast finishes the deployment of InsighFace on CPU/GPU and GPU accelerated by TensorRT. 

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory.

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted ArcFace model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# CPU inference
./infer_arcface_demo ms1mv3_arcface_r100.onnx face_0.jpg face_1.jpg face_2.jpg 0
# GPU inference
./infer_arcface_demo ms1mv3_arcface_r100.onnx face_0.jpg face_1.jpg face_2.jpg 1
# TensorRT inference on GPU
./infer_arcface_demo ms1mv3_arcface_r100.onnx face_0.jpg face_1.jpg face_2.jpg 2
```

The visualized result after running is as follows

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/en/faq/use_sdk_on_windows.md)

## InsightFace C++ Interface 

### ArcFace Class

```c++
fastdeploy::vision::faceid::ArcFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

ArcFace model loading and initialization, among which model_file is the exported ONNX model format.

### CosFace Class

```c++
fastdeploy::vision::faceid::CosFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

CosFace model loading and initialization, among which model_file is the exported ONNX model format.

### PartialFC Class

```c++
fastdeploy::vision::faceid::PartialFC(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

PartialFC model loading and initialization, among which model_file is the exported ONNX model format.

### VPL Class

```c++
fastdeploy::vision::faceid::VPL(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

VPL model loading and initialization, among which model_file is the exported ONNX model format.
**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Only passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict function

> ```c++
> ArcFace::Predict(const cv::Mat& im, FaceRecognitionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for FaceRecognitionResult

### Change pre-processing and post-processing parameters 
Pre-processing and post-processing parameters can be changed by modifying the member variables of InsightFaceRecognitionPostprocessor and InsightFaceRecognitionPreprocessor

#### InsightFaceRecognitionPreprocessor member variables (preprocessing parameters)
> > * **size**(vector&lt;int&gt;): This parameter changes the size of the resize during preprocessing, containing two integer elements for [width, height] with default value [112, 112].
      Revise through InsightFaceRecognitionPreprocessor::SetSize(std::vector<int>& size)
> > * **alpha**(vector&lt;float&gt;): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5].
      Revise through InsightFaceRecognitionPreprocessor::SetAlpha(std::vector<float>& alpha)
> > * **beta**(vector&lt;float&gt;): Preprocess normalized beta, and calculated as `x'=x*alpha+beta`. beta  defaults to [-1.f, -1.f, -1.f].
      Revise through InsightFaceRecognitionPreprocessor::SetBeta(std::vector<float>& beta)

#### InsightFaceRecognitionPostprocessor member variables (post-processing parameters)
> > * **l2_normalize**(bool): Whether to perform l2 normalization before outputting the face vector. Default false.
      Revise through InsightFaceRecognitionPostprocessor::SetL2Normalize(bool& l2_normalize)

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
