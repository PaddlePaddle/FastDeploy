# FastDeploy 一键模型量化
FastDeploy 给用户提供了一键量化功能, 支持离线量化和量化蒸馏训练. 本文档以Yolov5s为例, 供用户参考如何安装并执行FastDeploy的一键模型量化功能.

## 1.安装

### 环境依赖

1.用户参考PaddlePaddle官网, 安装develop版本
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.安装paddleslim-develop版本
```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

### FastDeploy-Quantization 安装方式
用户在当前目录下，运行如下命令:
```
python setup.py install
```

## 2.使用方式

### 一键量化示例

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

##### 2.使用fastdeploy_quant命令，执行一键模型量化:

```shell
fastdeploy_quant --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```

##### 3.参数说明

目前用户只需要提供一个定制的模型config文件,并指定量化方法和量化后的模型保存路径即可完成量化.

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键量化所需要的量化配置文件.[详解](./configs/README.md)                        |
| --method               | 量化方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |



#### 量化蒸馏训练

##### 1.准备待量化模型和训练数据集
FastDeploy目前的量化蒸馏训练，只支持无标注图片训练，训练过程中不支持评估模型精度.
数据集为真实预测场景下的图片，图片数量依据数据集大小来定，尽量覆盖所有部署场景. 此例中，我们为用户准备了COCO2017验证集中的前320张图片.
注: 如果用户想通过量化蒸馏训练的方法,获得精度更高的量化模型, 可以自行准备更多的数据, 以及训练更多的轮数.

```shell
# 下载yolov5.onnx
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO2017验证集中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2.使用fastdeploy_quant命令，执行一键模型量化:

```shell
# 执行命令默认为单卡训练，训练前请指定单卡GPU, 否则在训练过程中可能会卡住.
export CUDA_VISIBLE_DEVICES=0
fastdeploy_quant --config_path=./configs/detection/yolov5s_quant.yaml --method='QAT' --save_dir='./yolov5s_qat_model/'
```

##### 3.参数说明

目前用户只需要提供一个定制的模型config文件,并指定量化方法和量化后的模型保存路径即可完成量化.

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键量化所需要的量化配置文件.[详解](./configs/README.md)|
| --method               | 量化方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |


## 3. FastDeploy 部署量化模型
用户在获得量化模型之后，即可以使用FastDeploy进行部署, 部署文档请参考:
具体请用户参考示例文档:
- [YOLOv5 量化模型部署](../../examples/vision/detection/yolov5/quantize/)

- [YOLOv6 量化模型部署](../../examples/vision/detection/yolov6/quantize/)

- [YOLOv7 量化模型部署](../../examples/vision/detection/yolov7/quantize/)

- [PadddleClas 量化模型部署](../../examples/vision/classification/paddleclas/quantize/)
