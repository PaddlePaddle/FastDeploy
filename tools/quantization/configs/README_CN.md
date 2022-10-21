[English](README.md) | 简体中文

# FastDeploy 量化配置文件说明
FastDeploy 量化配置文件中，包含了全局配置，量化蒸馏训练配置，离线量化配置和训练配置.
用户除了直接使用FastDeploy提供在本目录的配置文件外，可以按需求自行修改相关配置文件

## 实例解读

```
# 全局配置
Global:
  model_dir: ./yolov5s.onnx                   #输入模型的路径
  format: 'onnx'                              #输入模型的格式, paddle模型请选择'paddle'
  model_filename: model.pdmodel               #量化后转为paddle格式模型的模型名字
  params_filename: model.pdiparams            #量化后转为paddle格式模型的参数名字
  image_path: ./COCO_val_320                  #离线量化或者量化蒸馏训练使用的数据集路径
  arch: YOLOv5                                #模型结构
  input_list: ['x2paddle_images']             #待量化的模型的输入名字
  preprocess: yolo_image_preprocess           #模型量化时,对数据做的预处理函数, 用户可以在 ../fdquant/dataset.py 中修改或自行编写新的预处理函数

#量化蒸馏训练配置
Distillation:
  alpha: 1.0                                  #蒸馏loss所占权重
  loss: soft_label                            #蒸馏loss算法

Quantization:
  onnx_format: true                           #是否采用ONNX量化标准格式, 要在FastDeploy上部署, 必须选true
  use_pact: true                              #量化训练是否使用PACT方法
  activation_quantize_type: 'moving_average_abs_max'     #激活量化方式
  quantize_op_types:                          #需要进行量化的OP
  - conv2d
  - depthwise_conv2d

#离线量化配置
PTQ:
  calibration_method: 'avg'                   #离线量化的激活校准算法, 可选: avg, abs_max, hist, KL, mse, emd
  skip_tensor_list: None                      #用户可指定跳过某些conv层,不进行量化

#训练参数配置
TrainConfig:
  train_iter: 3000
  learning_rate: 0.00001
  optimizer_builder:
    optimizer:
      type: SGD
    weight_decay: 4.0e-05
  target_metric: 0.365

```
## 更多详细配置方法

FastDeploy一键量化功能由PaddeSlim助力, 更详细的量化配置方法请参考:
[自动化压缩超参详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/hyperparameter_tutorial.md)
