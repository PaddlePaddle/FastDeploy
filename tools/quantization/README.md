[简体中文](README_CN.md) | English

# FastDeploy One-Click Quantization Tool

FastDeploy, based on PaddleSlim, provides developers with a one-click model quantization tool that supports post-training quantization and knowledge distillation training.
We take the Yolov5 series as an example to demonstrate how to install and execute FastDeploy's one-click model quantization.

## 1.Install

### Environment Dependencies

1. Install the develop version downloaded from PaddlePaddle official website.

```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.Install PaddleSlim-develop

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

### Install FastDeploy Quantization

Run the following command in the current directory

```
python setup.py install
```

## 2. How to Use

### Demo for One-Click Quantization Tool

#### Post-Training Quantization

##### 1. Prepare models and Calibration data set

Developers need to prepare the model to be quantized and the Calibration dataset on their own.
In this demo, developers can execute the following command to download the yolov5s.onnx model to be quantized and calibration data set. 

```shell
# Download yolov5.onnx
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx

# Download dataset. This Calibration dataset is the first 320 images from COCO val2017
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2. Run fastdeploy_quant command to quantize the model

The following command is to quantize the yolov5s model, if developers want to quantize other models, replace the config_path with other model configuration files in the configs folder.

```shell
fastdeploy_quant --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```
【说明】离线量化（训练后量化）：post-training quantization，缩写是PTQ

##### 3. Parameters

To complete the quantization, developers only need to provide a customized model config file, specify the quantization method, and the path to save the quantized model.

| Parameter     | Description                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| --config_path | Quantization profiles needed for one-click quantization [Configs](./configs/README.md)                        |
| --method      | Quantization method selection, PTQ for post-training quantization, QAT for quantization distillation training |
| --save_dir    | Output of quantized model paths, which can be deployed directly in FastDeploy                                 |

#### Quantized distillation training

##### 1.Prepare the model to be quantized and the training data set

FastDeploy currently supports quantized distillation training only for images without annotation. It does not support evaluating model accuracy during training.
The datasets are images from inference application, and the number of images is determined by the size of the dataset, covering all deployment scenarios as much as possible. In this demo, we prepare the first 320 images from the COCO2017 validation set for users.
Note: If users want to obtain a more accurate quantized model through quantized distillation training, feel free to prepare more data and train more rounds.

```shell
# Download yolov5.onnx
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx

# Download dataset. This Calibration dataset is the first 320 images from COCO val2017
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2.Use fastdeploy_quant command to quantize models

The following command is to quantize the yolov5s model, if developers want to quantize other models, replace the config_path with other model configuration files in the configs folder.

```shell
# Please specify the single card GPU before training, otherwise it may get stuck during the training process.
export CUDA_VISIBLE_DEVICES=0
fastdeploy_quant --config_path=./configs/detection/yolov5s_quant.yaml --method='QAT' --save_dir='./yolov5s_qat_model/'
```

##### 3.Parameters

To complete the quantization, developers only need to provide a customized model config file, specify the quantization method, and the path to save the quantized model.

| Parameter     | Description                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| --config_path | Quantization profiles needed for one-click quantization [Configs](./configs/README.md)                        |
| --method      | Quantization method selection, PTQ for post-training quantization, QAT for quantization distillation training |
| --save_dir    | Output of quantized model paths, which can be deployed directly in FastDeploy                                 |

## 3. Deploy quantized models on FastDeploy

Once obtained the quantized model, developers can deploy it on FastDeploy. Please refer to the following docs for more details

- [YOLOv5 Quantized Model Deployment](../../examples/vision/detection/yolov5/quantize/)

- [YOLOv6 Quantized Model Deployment](../../examples/vision/detection/yolov6/quantize/)

- [YOLOv7 Quantized Model Deployment](../../examples/vision/detection/yolov7/quantize/)

- [PadddleClas Quantized Model Deployment](../../examples/vision/classification/paddleclas/quantize/)
