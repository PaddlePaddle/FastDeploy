# FastDeploy 量化配置文件说明
FastDeploy 量化配置文件中，包含了全局配置，量化蒸馏训练配置，离线量化配置和训练配置.
用户除了直接使用FastDeploy提供在本目录的配置文件外，可以按需求自行修改相关配置文件

## 实例解读

```
#全局信息
Global:
  model_dir: ./yolov7-tiny.onnx     #输入模型路径
  format: 'onnx'                    #输入模型格式，选项为 onnx 或者 paddle
  model_filename: model.pdmodel     #paddle模型的模型文件名
  params_filename: model.pdiparams  #paddle模型的参数文件名
  image_path: ./COCO_val_320        #PTQ所有的Calibration数据集或者量化训练所用的训练集
  arch: YOLOv7                      #模型系列

#量化蒸馏训练中的蒸馏参数设置
Distillation:
  alpha: 1.0
  loss: soft_label

#量化蒸馏训练中的量化参数设置
Quantization:
  onnx_format: true
  activation_quantize_type: 'moving_average_abs_max'
  quantize_op_types:
  - conv2d
  - depthwise_conv2d

#离线量化参数配置
PTQ:
  calibration_method: 'avg' #Calibraion算法，可选为 avg, abs_max, hist, KL, mse
  skip_tensor_list: None    #不进行离线量化的tensor


#训练参数
TrainConfig:
  train_iter: 3000  
  learning_rate:
    type: CosineAnnealingDecay
    learning_rate: 0.00003
    T_max: 8000
  optimizer_builder:
    optimizer:
      type: SGD
    weight_decay: 0.00004

```
