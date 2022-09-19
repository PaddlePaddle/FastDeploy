# FastDeploy 一键模型量化
FastDeploy 给用户提供了一键量化功能, 用户可以参考本文档安装FastDeploy一键量化功能.

## 1.安装

### 环境依赖
- python >= 3.5  
- paddlepaddle >= 2.3 (如需使用GPU进行量化，请下载GPU版本)
- paddleslim >= 2.3.3

### 安装方式
用户在当前目录下，运行如下命令:
```
python setup.py install
```

## 2.使用方式

### 一键量化示例

#### 1. 准备待量化模型与Calibration数据集

用户需要自行准备待量化模型与Calibration数据集.
本例中用户可执行以下命令, 下载待量化的yolov5s.onnx模型和我们为用户准备的Calibration数据集示例.

```shell
# 下载yolov5.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO val2017中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

#### 2.使用fdquant命令，执行一键模型量化:

```shell
fdquant --model_type=YOLOv5 --model_file=./yolov5s.onnx --save_dir=./yolov5s_quant_out --data_dir=./COCO_val_320 --calibration_method=avg
```
注意: 在执行以上命令时, 如果遇到input shape检查提示, 用户请直接输入'N'跳过即可.

### 参数说明

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --model_type          | 待量化的模型系列, 上例为YOLOv5系列模型                         |
| --model_file           | 输入的模型文件, 上例为yolov5s.onnx     |
| --save_dir             | 产出的量化后模型路径, 上例为yolov5s_quant_out      |
| --data_dir             | 用于模型量化的calibration数据集, 用户自行准备验证集中的少量数据即可，默认为320张图片|
| --calibration_method   | 离线量化中，用于对activation层获取量化信息的方法, 用户可选: avg, max_abs, hist, mse, KL. 默认为avg|

注意：目前fdquant暂时只支持YOLOv5,YOLOv6和YOLOv7模型的量化

## 3. FastDeploy 部署量化模型
用户在获得量化模型之后，只需要简单地传入量化后的模型路径及相应参数，即可以使用FastDeploy进行部署.
具体请用户参考示例文档:
- [YOLOv5s 量化模型Python部署](../examples/slim/yolov5s/python/)
- [YOLOv5s 量化模型C++部署](../examples/slim/yolov5s/cpp/)
- [YOLOv6s 量化模型Python部署](../examples/slim/yolov6s/python/)
- [YOLOv6s 量化模型C++部署](../examples/slim/yolov6s/cpp/)
- [YOLOv7 量化模型Python部署](../examples/slim/yolov7/python/)
- [YOLOv7 量化模型C++部署](../examples/slim/yolov7/cpp/)
