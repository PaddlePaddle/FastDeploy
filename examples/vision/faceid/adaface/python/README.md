English | [简体中文](README_CN.md)
# AdaFace Python Deployment Example
This directory provides examples that `infer_xxx.py` fast finishes the deployment of AdaFace on CPU/GPU and GPU accelerated by TensorRT.

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy  Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Taking AdaFace as an example, we demonstrate how `infer.py` fast finishes the deployment of AdaFace on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/faceid/adaface/python/

# Download AdaFace model files and test images
# Download test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# Run the following code if the model is in Paddle format
wget https://bj.bcebos.com/paddlehub/fastdeploy/mobilefacenet_adaface.tgz
tar zxvf mobilefacenet_adaface.tgz -C ./

# CPU inference
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device cpu
# GPU inference
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device gpu
# TensorRT inference on GPU
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face face_0.jpg \
                --face_positive face_1.jpg \
                --face_negative face_2.jpg \
                --device gpu \
                --use_trt True

# KunlunXin XPU inference
python infer.py --model mobilefacenet_adaface/mobilefacenet_adaface.pdmodel \
                --params_file mobilefacenet_adaface/mobilefacenet_adaface.pdiparams \
                --face test_lite_focal_arcface_0.JPG \
                --face_positive test_lite_focal_arcface_1.JPG \
                --face_negative test_lite_focal_arcface_2.JPG \
                 --device kunlunxin
```

The visualized result after running is as follows

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

```bash
FaceRecognitionResult: [Dim(512), Min(-0.133213), Max(0.148838), Mean(0.000293)]
FaceRecognitionResult: [Dim(512), Min(-0.102777), Max(0.120130), Mean(0.000615)]
FaceRecognitionResult: [Dim(512), Min(-0.116685), Max(0.142919), Mean(0.001595)]
Cosine 01:  0.7483505506964364
Cosine 02:  -0.09605773855893639
```

## AdaFace Python Interface 

```python
fastdeploy.vision.faceid.AdaFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.PADDLE)
```

AdaFace  model loading and initialization, among which model_file is the exported ONNX model format or PADDLE static graph format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> AdaFace.predict(image_data)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.FaceRecognitionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

#### Member variables of AdaFacePreprocessor
The member variables of AdaFacePreprocessor are as follows
> > * **size**(list[int]): This parameter changes the size of the resize during preprocessing, containing two integer elements for [width, height] with default value [112, 112]
> > * **alpha**(list[float]): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(list[float]): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. beta defaults to [-1.f, -1.f, -1.f]
> > * **swap_rb**(bool): Whether to convert BGR to RGB in pre-processing. Default true

#### Member variables of AdaFacePostprocessor
The member variables of AdaFacePostprocessor are as follows
> > * **l2_normalize**(bool): Whether to perform l2 normalization before outputting the face vector. Default false.


## Other Documents

- [AdaFace Model Description](..)
- [AdaFace C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
