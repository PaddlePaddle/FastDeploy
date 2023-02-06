English | [简体中文](README_CN.md)
# InsightFace C++ Deployment Example

FastDeploy supports the deployment of InsightFace models like ArcFace\CosFace\VPL\Partial_FC on RKNPU.

This directoty provides the example that `infer_arcface.cc` fast finishes the deployment of InsighFace models like ArcFace on CPU/RKNPU.


Two steps before deployment:

1. Software and hardware should meet the requirements. 
2. Download the precompiled deployment library or deploy FastDeploy repository from scratch according to your development environment. 

Refer to [RK2 generation NPU deployment library compilation](../../../../../../docs/cn/build_and_install/rknpu2.md) for the above steps

The compilation can be completed by executing the following command in this directory. 

```bash
mkdir build
cd build
# FastDeploy version need >=1.0.3
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the official converted ArcFace model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r18.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# CPU inference
./infer_arcface_demo ms1mv3_arcface_r100.onnx face_0.jpg face_1.jpg face_2.jpg 0
# RKNPU inference
./infer_arcface_demo ms1mv3_arcface_r100.onnx face_0.jpg face_1.jpg face_2.jpg 1
```

The visualized result is as follows

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

The above command works for Linux or MacOS. For SDK in Windows, refer to: 
- [How to use FastDeploy C++ SDK in Windows](../../../../../../docs/cn/faq/use_sdk_on_windows.md)

## InsightFace C++ Interface

### ArcFace 

```c++
fastdeploy::vision::faceid::ArcFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

ArcFace model loading and initialization, among which model_file is the exported ONNX model format

### CosFace

```c++
fastdeploy::vision::faceid::CosFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

CosFace model loading and initialization, among which model_file is the exported ONNX model format

### PartialFC

```c++
fastdeploy::vision::faceid::PartialFC(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

PartialFC model loading and initialization, among which model_file is the exported ONNX model format

### VPL

```c++
fastdeploy::vision::faceid::VPL(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

VPL model loading and initialization, among which model_file is the exported ONNX model format
**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. Merely passing an empty string when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

#### Predict function

> ```c++
> ArcFace::Predict(const cv::Mat& im, FaceRecognitionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results] for the description of FaceRecognitionResult(../../../../../../docs/api/vision_results/)

### Change pre-processing and post-processing parameters 
Pre-processing and post-processing parameters can be changed by modifying the member variables of InsightFaceRecognitionPostprocessor and InsightFaceRecognitionPreprocessor

#### Member variables of InsightFaceRecognitionPreprocessor (preprocessing parameters)
> > * **size**(vector&lt;int&gt;): This parameter changes the resize during preprocessing, containing two integer elements for [width, height] with default value [112, 112].
      Revise through InsightFaceRecognitionPreprocessor::SetSize(std::vector<int>& size)
> > * **alpha**(vector&lt;float&gt;): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. Alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5].
      Revise through InsightFaceRecognitionPreprocessor::SetAlpha(std::vector<float>& alpha)
> > * **beta**(vector&lt;float&gt;): Preprocess normalized beta, and calculated as `x'=x*alpha+beta`. Alpha defaults to [-1.f, -1.f, -1.f],
      Revise through InsightFaceRecognitionPreprocessor::SetBeta(std::vector<float>& beta)

####  Member variables of InsightFaceRecognitionPostprocessor(post-processing parameters)
> > * **l2_normalize**(bool): Whether to perform l2 normalization before outputting the face vector. Default false.
      Revise through InsightFaceRecognitionPostprocessor::SetL2Normalize(bool& l2_normalize)

- [Model Description](../../../)
- [Python Deployemnt](../python)
- [Vision Model Prediction Results](../../../../../../docs/api/vision_results/README.md)
- [How to switch the backend engine](../../../../../../docs/cn/faq/how_to_change_backend.md)
