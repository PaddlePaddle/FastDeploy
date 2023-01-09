English | [简体中文](README_CN.md)
# FastestDet Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of FastestDet on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/fastestdet/python/

# Download fastestdet model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/FastestDet.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU inference
python infer.py --model FastestDet.onnx --image 000000014439.jpg --device cpu
# GPU inference
python infer.py --model FastestDet.onnx --image 000000014439.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model FastestDet.onnx --image 000000014439.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/44280887/206176291-61eb118b-391b-4431-b79e-a393b9452138.jpg">

## FastestDet Python Interface 

```python
fastdeploy.vision.detection.FastestDet(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

FastestDet model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> FastestDet.predict(image_data)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.DetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its structure

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize used during preprocessing, containing two integer elements for [width, height] with default value [352, 352]


## Other Documents

- [FastestDet Model Description](..)
- [FastestDet C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
