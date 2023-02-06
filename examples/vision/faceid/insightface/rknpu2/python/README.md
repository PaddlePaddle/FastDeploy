English | [简体中文](README_CN.md)
# InsightFace Python Deployment Example

FastDeploy supports the deployment of InsightFace models like ArcFace\CosFace\VPL\Partial on RKNPU.

This directoty provides the example that `infer_arcface.py` fast finishes the deployment of InsighFace models like ArcFace on CPU/RKNPU.


Two steps before deployment:

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/cn/build_and_install/rknpu2.md)

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/faceid/insightface/python/

# Download ArcFace model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ms1mv3_arcface_r100.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/face_demo.zip
unzip face_demo.zip

# CPU inference
python infer_arcface.py --model ms1mv3_arcface_r100.onnx \
                        --face face_0.jpg \
                        --face_positive face_1.jpg \
                        --face_negative face_2.jpg \
                        --device cpu
# GPU inference
python infer_arcface.py --model ms1mv3_arcface_r100.onnx \
                        --face face_0.jpg \
                        --face_positive face_1.jpg \
                        --face_negative face_2.jpg \
                        --device gpu
```

The visualized result is as follows

<div width="700">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321537-860bf857-0101-4e92-a74c-48e8658d838c.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184322004-a551e6e4-6f47-454e-95d6-f8ba2f47b516.JPG">
<img width="220" float="left" src="https://user-images.githubusercontent.com/67993288/184321622-d9a494c3-72f3-47f1-97c5-8a2372de491f.JPG">
</div>

```bash
Prediction Done!
--- [Face 0]:FaceRecognitionResult: [Dim(512), Min(-2.309220), Max(2.372197), Mean(0.016987)]
--- [Face 1]:FaceRecognitionResult: [Dim(512), Min(-2.288258), Max(1.995104), Mean(-0.003400)]
--- [Face 2]:FaceRecognitionResult: [Dim(512), Min(-3.243411), Max(3.875866), Mean(-0.030682)]
Detect Done! Cosine 01: 0.814385, Cosine 02:-0.059388

```

## InsightFace Python interface

```python
fastdeploy.vision.faceid.ArcFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
fastdeploy.vision.faceid.CosFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
fastdeploy.vision.faceid.PartialFC(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
fastdeploy.vision.faceid.VPL(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

ArcFace model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> ArcFace.predict(image_data)
> ```
>
> Model prediction interface. Input images and output prediction results
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return the `fastdeploy.vision.FaceRecognitionResult` structure. Refer to [Vision Model Prediction Results](../../../../../../docs/api/vision_results/) for its description

### Class Member Property
#### Pre-processing Parameter
Users can modify the following preprocessing parameters based on actual needs to change the final inference and deployment results.

#### Member Variables of AdaFacePreprocessor
The followings are the member variables of AdaFacePreprocessor
> > * **size**(list[int]): This parameter changes the resize used during preprocessing, containing two integer elements for [width, height] with default value [112, 112]
> > * **alpha**(list[float]): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. Alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(list[float]): Preprocess normalized beta, and calculated as `x'=x*alpha+beta`. beta defaults to [-1.f, -1.f, -1.f]

#### Member Variables of AdaFacePostprocessor
The followings are the member variables of AdaFacePostprocessor
> > * **l2_normalize**(bool): Whether to perform l2 normalization before outputting the face vector. Default false.


## Other Documents

- [InsightFace Model Description](..)
- [InsightFace C++ Deployment](../cpp)
- [Vision Model Prediction Results](../../../../../../docs/api/vision_results/)
- [How to switch the backend engine](../../../../../../docs/cn/faq/how_to_change_backend.md)
