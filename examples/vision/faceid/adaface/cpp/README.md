English | [简体中文](README_CN.md)
# AdaFace C++ Deployment Example
This directory provides examples that `infer_xxx.py` fast finishes the deployment of AdaFace on CPU/GPU and GPU accelerated by TensorRT. 

Taking AdaFace as an example, we demonstrate how `infer.cc` fast finishes the deployment of AdaFace on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking the CPU inference on Linux as an example, the compilation test can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.

```bash
# “If the precompiled library does not contain this model, compile SDK from the latest code”
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# Run the following code if the model is in Paddle format
wget https://bj.bcebos.com/paddlehub/fastdeploy/mobilefacenet_adaface.tgz
tar zxvf mobilefacenet_adaface.tgz -C ./
# CPU inference
./infer_adaface_demo mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
              mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
              face_0.jpg face_1.jpg face_2.jpg 0

# GPU inference
./infer_adaface_demo mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
              mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
              face_0.jpg face_1.jpg face_2.jpg 1

# GPU上TensorRT推理
./infer_adaface_demo mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
              mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
              face_0.jpg face_1.jpg face_2.jpg 2

# KunlunXin XPU inference
./infer_demo mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
              mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
              face_0.jpg face_1.jpg face_2.jpg 3
```

The visualized result after running is as follows

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

The above command works for Linux or MacOS. For SDK use-pattern in Windows, refer to:
- [How to use FastDeploy C++ SDK in Windows](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## AdaFace C++ Interface 

### AdaFace Class 

```c++
fastdeploy::vision::faceid::AdaFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::PADDLE)
```

AdaFace model loading and initialization, model_file and params_file are in PaddleInference format if using PaddleInference for inference;
model_file is in ONNX format and params_file is empty if using ONNXRuntime for inference


#### Predict Function

> ```c++
> AdaFace::Predict(cv::Mat* im, FaceRecognitionResult* result)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **im**: Input images in HWC or BGR format
> > * **result**: Detection results, including detection box and confidence of each box. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for FaceRecognitionResult.

### Revise pre-processing and post-processing parameters 
Pre-processing and post-processing parameters can be changed by modifying the member variables of AdaFacePostprocessor and AdaFacePreprocessor.

#### AdaFacePreprocessor member variables (preprocessing parameters)
> > * **size**(vector&lt;int&gt;): This parameter changes the size of the resize during preprocessing, containing two integer elements for [width, height] with default value [112, 112].
      Revise through AdaFacePreprocessor::SetSize(std::vector<int>& size)
> > * **alpha**(vector&lt;float&gt;): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5].
      Revise through AdaFacePreprocessor::SetAlpha(std::vector<float>& alpha)
> > * **beta**(vector&lt;float&gt;): Preprocess normalized beta, and calculated as `x'=x*alpha+beta`，beta defaults to [-1.f, -1.f, -1.f],
      Revise through AdaFacePreprocessor::SetBeta(std::vector<float>& beta)
> > * **permute**(bool): Whether to convert BGR to RGB in pre-processing. Default true.
      Revise through AdaFacePreprocessor::SetPermute(bool permute)

#### AdaFacePostprocessor member variables (post-processing parameters)
> > * **l2_normalize**(bool): Whether to perform l2 normalization before outputting the face vector. Default false.
      Revise through AdaFacePostprocessor::SetL2Normalize(bool& l2_normalize)

- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
