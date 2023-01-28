English | [简体中文](README_CN.md)
# YOLOv5Cls Python Deployment Example

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md). 
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

This directory provides examples that `infer.py` fast finishes the deployment of YOLOv5Cls on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/classification/yolov5cls/python/

# Download the YOLOv5Cls model file and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.onnx
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU inference
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1
# GPU inference
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --topk 1
# TensorRT inference on GPU 
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True
```

The result returned after running is as follows
```bash
ClassifyResult(
label_ids: 265,
scores: 0.196327,
)
```

## YOLOv5Cls Python Interface 

```python
fastdeploy.vision.classification.YOLOv5Cls(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv5Cls model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict Function

> ```python
> YOLOv5Cls.predict(image_data, topk=1)
> ```
>
> Model prediction interface. Input images and output classification topk results directly.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format
> > * **topk**(int): Return the topk classification results with the highest prediction probability. Default 1

> **Return**
>
> > Return `fastdeploy.vision.ClassifyResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.


## Other Documents

- [YOLOv5Cls Model Description](..)
- [YOLOv5Cls C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
