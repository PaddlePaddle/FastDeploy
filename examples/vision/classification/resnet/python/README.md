English | [简体中文](README_CN.md)
# ResNet Model Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

This directory provides examples that `infer.py` fast finishes the deployment of ResNet50_vd on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/resnet/python

# Download the ResNet50_vd model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/resnet50.onnx
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU inference
python infer.py --model resnet50.onnx --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1
# GPU inference
python infer.py --model resnet50.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --topk 1
# Use TensorRT inference on GPU  （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model resnet50.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1
```

The result returned after running is as follows
```bash
ClassifyResult(
label_ids: 332,
scores: 0.825349,
)
```

## ResNet Python Interface 

```python
fd.vision.classification.ResNet(model_file, params_file, runtime_option=None, model_format=ModelFormat.ONNX)
```


**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path 
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict Function

> ```python
> ResNet.predict(input_image, topk=1)
> ```
>
> Model prediction interface. Input images and output results directly.
>
> **parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format
> > * **topk**(int): Return the topk classification results with the highest prediction probability. Default 1

> **Return**
>
> > Return `fastdeploy.vision.ClassifyResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.


## Other Documents

- [ResNet Model Description](..)
- [ResNet C++ Deployment](../cpp)
- [Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
