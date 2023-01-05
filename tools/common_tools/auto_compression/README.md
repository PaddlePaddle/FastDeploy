# FastDeploy 一键模型自动化压缩
FastDeploy基于PaddleSlim的Auto Compression Toolkit(ACT), 给用户提供了一键模型自动化压缩的工具.
本文档以Yolov5s为例, 供用户参考如何安装并执行FastDeploy的一键模型自动化压缩.

## 1.安装

### 环境依赖

1.用户参考PaddlePaddle官网, 安装Paddle 2.4 版本
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.安装PaddleSlim 2.4 版本
```bash
pip install paddleslim==2.4.0
```

### 一键模型自动化压缩工具安装方式
FastDeploy一键模型自动化压缩不需要单独的安装, 用户只需要正确安装好[FastDeploy工具包](../../README.md)即可.

## 2.使用方式

### 一键模型压缩示例
FastDeploy模型一键自动压缩可包含多种策略, 目前主要采用离线量化和量化蒸馏训练, 下面将从离线量化和量化蒸馏两个策略来介绍如何使用一键模型自动化压缩.

#### 离线量化

##### 1. 准备模型和Calibration数据集
用户需要自行准备待量化模型与Calibration数据集.
本例中用户可执行以下命令, 下载待量化的yolov5s.onnx模型和我们为用户准备的Calibration数据集示例.

```shell
# 下载yolov5.onnx
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO val2017中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2.使用fastdeploy compress命令，执行一键模型自动化压缩:
以下命令是对yolov5s模型进行量化, 用户若想量化其他模型, 替换config_path为configs文件夹下的其他模型配置文件即可.
```shell
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```

##### 3.参数说明

目前用户只需要提供一个定制的模型config文件,并指定量化方法和量化后的模型保存路径即可完成量化.

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键压缩所需要的量化配置文件.[详解](./configs/README.md)                        |
| --method               | 压缩方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |



#### 量化蒸馏训练

##### 1.准备待量化模型和训练数据集
FastDeploy一键模型自动化压缩目前的量化蒸馏训练，只支持无标注图片训练，训练过程中不支持评估模型精度.
数据集为真实预测场景下的图片，图片数量依据数据集大小来定，尽量覆盖所有部署场景. 此例中，我们为用户准备了COCO2017训练集中的前320张图片.
注: 如果用户想通过量化蒸馏训练的方法,获得精度更高的量化模型, 可以自行准备更多的数据, 以及训练更多的轮数.

```shell
# 下载yolov5.onnx
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO2017训练集中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_train_320.tar
tar -xvf COCO_train_320.tar
```

##### 2.使用fastdeploy compress命令，执行一键模型自动化压缩:
以下命令是对yolov5s模型进行量化, 用户若想量化其他模型, 替换config_path为configs文件夹下的其他模型配置文件即可.
```shell
# 执行命令默认为单卡训练，训练前请指定单卡GPU, 否则在训练过程中可能会卡住.
export CUDA_VISIBLE_DEVICES=0
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='QAT' --save_dir='./yolov5s_qat_model/'
```

##### 3.参数说明

目前用户只需要提供一个定制的模型config文件,并指定量化方法和量化后的模型保存路径即可完成量化.

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键自动化压缩所需要的量化配置文件.[详解](./configs/README.md)|
| --method               | 压缩方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |


## 3. FastDeploy 一键模型自动化压缩 Config文件参考
FastDeploy目前为用户提供了多个模型的压缩[config](./configs/)文件,以及相应的FP32模型, 用户可以直接下载使用并体验.

| Config文件                | 待压缩的FP32模型 | 备注                                                       |
| -------------------- | ------------------------------------------------------------ |----------------------------------------- |
| [mobilenetv1_ssld_quant](./configs/classification/mobilenetv1_ssld_quant.yaml)      | [mobilenetv1_ssld](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV1_ssld_infer.tgz)           |           |
| [resnet50_vd_quant](./configs/classification/resnet50_vd_quant.yaml)      |   [resnet50_vd](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz)          |     |
| [efficientnetb0_quant](./configs/classification/efficientnetb0_quant.yaml)      |   [efficientnetb0](https://bj.bcebos.com/paddlehub/fastdeploy/EfficientNetB0_small_infer.tgz)          |     |
| [mobilenetv3_large_x1_0_quant](./configs/classification/mobilenetv3_large_x1_0_quant.yaml)      |   [mobilenetv3_large_x1_0](https://bj.bcebos.com/paddlehub/fastdeploy/MobileNetV3_large_x1_0_ssld_infer.tgz)          |     |
| [pphgnet_tiny_quant](./configs/classification/pphgnet_tiny_quant.yaml)      |   [pphgnet_tiny](https://bj.bcebos.com/paddlehub/fastdeploy/PPHGNet_tiny_ssld_infer.tgz)          |     |
| [pplcnetv2_base_quant](./configs/classification/pplcnetv2_base_quant.yaml)      |   [pplcnetv2_base](https://bj.bcebos.com/paddlehub/fastdeploy/PPLCNetV2_base_infer.tgz)          |     |
| [yolov5s_quant](./configs/detection/yolov5s_quant.yaml)       |   [yolov5s](https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx)         |     |
| [yolov6s_quant](./configs/detection/yolov6s_quant.yaml)       |  [yolov6s](https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx)          |     |
| [yolov7_quant](./configs/detection/yolov7_quant.yaml)        | [yolov7](https://paddle-slim-models.bj.bcebos.com/act/yolov7.onnx)           |      |
| [ppyoloe_withNMS_quant](./configs/detection/ppyoloe_withNMS_quant.yaml)       |  [ppyoloe_l](https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar)    | 支持PPYOLOE的s,m,l,x系列模型, 从PaddleDetection导出模型时正常导出, 不要去除NMS |
| [ppyoloe_plus_withNMS_quant](./configs/detection/ppyoloe_plus_withNMS_quant.yaml)       |  [ppyoloe_plus_s](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_plus_crn_s_80e_coco.tar)    | 支持PPYOLOE+的s,m,l,x系列模型, 从PaddleDetection导出模型时正常导出, 不要去除NMS |
| [pp_liteseg_quant](./configs/segmentation/pp_liteseg_quant.yaml)    |   [pp_liteseg](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer.tgz)        |
| [deeplabv3_resnet_quant](./configs/segmentation/deeplabv3_resnet_quant.yaml)    |   [deeplabv3_resnet101](https://bj.bcebos.com/paddlehub/fastdeploy/Deeplabv3_ResNet101_OS8_cityscapes_without_argmax_infer.tgz)        |       |
| [fcn_hrnet_quant](./configs/segmentation/fcn_hrnet_quant.yaml)    |   [fcn_hrnet](https://bj.bcebos.com/paddlehub/fastdeploy/FCN_HRNet_W18_cityscapes_without_argmax_infer.tgz)        |       |
| [unet_quant](./configs/segmentation/unet_quant.yaml)    |   [unet](https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz)        |       |      |



## 4. FastDeploy 部署量化模型
用户在获得量化模型之后，即可以使用FastDeploy进行部署, 部署文档请参考:
具体请用户参考示例文档:
- [YOLOv5 量化模型部署](../../../examples/vision/detection/yolov5/quantize/)

- [YOLOv6 量化模型部署](../../../examples/vision/detection/yolov6/quantize/)

- [YOLOv7 量化模型部署](../../../examples/vision/detection/yolov7/quantize/)

- [PadddleClas 量化模型部署](../../../examples/vision/classification/paddleclas/quantize/)

- [PadddleDetection 量化模型部署](../../../examples/vision/detection/paddledetection/quantize/)

- [PadddleSegmentation 量化模型部署](../../../examples/vision/segmentation/paddleseg/quantize/)
