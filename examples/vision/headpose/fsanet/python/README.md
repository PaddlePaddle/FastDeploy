English | [简体中文](README_CN.md)
# FSANet Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of FSANet on CPU/GPU and GPU accelerated by TensorRT. FastDeploy version 0.6.0 or above is required to support this model. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/headpose/fsanet/python

# Download the FSANet model files and test images
## Original ONNX Model
wget https://bj.bcebos.com/paddlehub/fastdeploy/fsanet-var.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/headpose_input.png
# CPU inference
python infer.py --model fsanet-var.onnx --image headpose_input.png --device cpu
# GPU inference
python infer.py --model fsanet-var.onnx --image headpose_input.png --device gpu
# TRT inference
python infer.py --model fsanet-var.onnx --image headpose_input.png --device gpu --backend trt
```

The visualized result after running is as follows

<div width="520">
<img width="500" height="514" float="left" src="https://user-images.githubusercontent.com/19977378/198279932-3eee424e-98a2-4249-bdeb-0f79127cbc9d.png">
</div>

## FSANet Python Interface 

```python
fd.vision.headpose.FSANet(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

FSANet model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default
### predict function

> ```python
> FSANet.predict(input_image)
> ```
>
> Model prediction interface. Input images and output head pose prediction results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format
> **Return**
>
> > Return `fastdeploy.vision.HeadPoseResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure

## Other Documents

- [FSANet Model Description](..)
- [FSANet C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
