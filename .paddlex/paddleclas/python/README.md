English | [简体中文](README_CN.md)
# Example of PaddleClas models Python Deployment

```bash
# Find the model directory in the package, e.g. ResNet50

# Prepare a test image, e.g. test.jpg

# CPU inference
python infer.py --model ResNet50 --image test.jpg --device cpu --topk 1
# GPU inference
python infer.py --model ResNet50 --image test.jpg --device gpu --topk 1
# Use TensorRT inference on GPU （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model ResNet50 --image test.jpg --device gpu --use_trt True --topk 1
# IPU inference（Attention: It is somewhat time-consuming for the operation of model serialization when running IPU inference for the first time. Please be patient.）
python infer.py --model ResNet50 --image test.jpg --device ipu --topk 1
# XPU inference
python infer.py --model ResNet50 --image test.jpg --device xpu --topk 1
```

## PaddleClasModel Python Interface

```python
fd.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

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
> > Return `fastdeploy.vision.ClassifyResult` structure. Refer to [Visual Model Prediction Results](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result.md) for the description of the structure.
