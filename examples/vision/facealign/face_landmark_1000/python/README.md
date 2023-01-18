English | [简体中文](README_CN.md)
# FaceLandmark1000 Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy  Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of FaceLandmark1000 models on CPU/GPU and GPU accelerated by TensorRT. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/facealign/facelandmark1000/python

# Download the FaceLandmark1000 model file and test images 
## Original ONNX Model
wget https://bj.bcebos.com/paddlehub/fastdeploy/FaceLandmark1000.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/facealign_input.png

# CPU inference
python infer.py --model FaceLandmark1000.onnx --image facealign_input.png --device cpu
# GPU inference
python infer.py --model FaceLandmark1000.onnx --image facealign_input.png --device gpu
# TRT inference
python infer.py --model FaceLandmark1000.onnx --image facealign_input.png --device gpu --backend trt
```

The visualized result after running is as follows

<div width="500">
<img width="470" height="384" float="left" src="https://user-images.githubusercontent.com/67993288/200761309-90c096e2-c2f3-4140-8012-32ed84e5f389.jpg">
</div>

## FaceLandmark1000 Python Interface 

```python
fd.vision.facealign.FaceLandmark1000(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

FaceLandmark1000 model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict Function

> ```python
> FaceLandmark1000.predict(input_image)
> ```
>
> Model prediction interface. Input images and output landmarks results directly.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format
> **Return**
>
> > Return `fastdeploy.vision.FaceAlignmentResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure


## Other Documents

- [FaceLandmark1000 Model Description](..)
- [FaceLandmark1000 C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
