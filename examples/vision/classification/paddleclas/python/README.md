English | [简体中文](README_CN.md)
# Example of PaddleClas models Python Deployment

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install the FastDeploy Python whl package. Please refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of ResNet50_vd on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/python

# Download the ResNet50_vd model file and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU inference
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1
# GPU inference
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --topk 1
# Use TensorRT inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1
# IPU inference（Attention: It is somewhat time-consuming for the operation of model serialization when running IPU inference for the first time. Please be patient.）
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device ipu --topk 1
# XPU inference
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device xpu --topk 1
```

The result returned after running is as follows
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## PaddleClasModel Python接口

```python
fd.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleClas model loading and initialization, where model_file and params_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA) for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path 
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend Inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> PaddleClasModel.predict(input_image, topk=1)
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
> > Return `fastdeploy.vision.ClassifyResult` structure. Refer to [Visual Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.


## Other documents

- [PaddleClas Model Description](..)
- [PaddleClas C++ Deployment](../cpp)
- [Model prediction results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
