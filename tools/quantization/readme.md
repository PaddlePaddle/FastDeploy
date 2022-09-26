# FastDeploy 一键模型量化
FastDeploy 给用户提供了一键量化功能, 支持离线量化和量化蒸馏训练, 用户可以参考本文档安装FastDeploy一键量化功能.

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

### 一键离线量化示例

#### 离线量化

##### 1. 准备模型和Calibration数据集
用户需要自行准备待量化模型与Calibration数据集.
本例中用户可执行以下命令, 下载待量化的yolov5s.onnx模型和我们为用户准备的Calibration数据集示例.

```shell
# 下载yolov5.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO val2017中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2.使用fastdeploy_quant命令，执行一键模型量化:

```shell
fastdeploy_quant --config_path=./configs/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```

##### 3.参数说明

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键量化所需要的量化配置文件.[详解](./fdquant/configs/readme.md)                        |
| --method               | 量化方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |

注意：目前fastdeploy_quant暂时只支持YOLOv5,YOLOv6和YOLOv7模型的量化


#### 量化蒸馏训练

##### 1.准备待量化模型和训练数据集
FastDeploy目前的量化蒸馏训练，只支持无标注图片训练，训练过程中不支持评估模型精度.
数据集为真实预测场景下的图片，图片数量依据数据集大小来定，尽量覆盖所有部署场景. 此例中，我们为用户准备了COCO2017验证集中的前320张图片.

```shell
# 下载yolov5.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

# 下载数据集, 此Calibration数据集为COCO2017验证集中的前320张图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/COCO_val_320.tar.gz
tar -xvf COCO_val_320.tar.gz
```

##### 2.使用fastdeploy_quant命令，执行一键模型量化:

```shell
export CUDA_VISIBLE_DEVICES=0
fastdeploy_quant --config_path=./configs/yolov5s_quant.yaml --method='QAT' --save_dir='./yolov5s_qat_model/'
```

##### 3.参数说明

| 参数                 | 作用                                                         |
| -------------------- | ------------------------------------------------------------ |
| --config_path          | 一键量化所需要的量化配置文件.[详解](./fdquant/configs/readme.md)                        |
| --method               | 量化方式选择, 离线量化选PTQ，量化蒸馏训练选QAT     |
| --save_dir             | 产出的量化后模型路径, 该模型可直接在FastDeploy部署     |

注意：目前fastdeploy_quant暂时只支持YOLOv5,YOLOv6和YOLOv7模型的量化


## 3. FastDeploy 部署量化模型
用户在获得量化模型之后，只需要简单地传入量化后的模型路径及相应参数，即可以使用FastDeploy进行部署.
具体请用户参考示例文档:
- [YOLOv5s 量化模型Python部署](../examples/slim/yolov5s/python/)
- [YOLOv5s 量化模型C++部署](../examples/slim/yolov5s/cpp/)
- [YOLOv6s 量化模型Python部署](../examples/slim/yolov6s/python/)
- [YOLOv6s 量化模型C++部署](../examples/slim/yolov6s/cpp/)
- [YOLOv7 量化模型Python部署](../examples/slim/yolov7/python/)
- [YOLOv7 量化模型C++部署](../examples/slim/yolov7/cpp/)
