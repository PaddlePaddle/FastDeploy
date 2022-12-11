# FastDeploy 一键自动化压缩配置文件说明
FastDeploy 一键自动化压缩配置文件中，包含了全局配置，量化蒸馏训练配置，离线量化配置和训练配置.
用户除了直接使用FastDeploy提供在本目录的配置文件外，可以按照以下示例,自行修改相关配置文件, 来尝试压缩自己的模型.

## 实例解读

```
# 全局配置
Global:
  model_dir: ./ppyoloe_plus_crn_s_80e_coco    #输入模型的路径, 用户若需量化自己的模型，替换此处即可
  format: paddle                              #输入模型的格式, paddle模型请选择'paddle', onnx模型选择'onnx'
  model_filename: model.pdmodel               #量化后转为paddle格式模型的模型名字
  params_filename: model.pdiparams            #量化后转为paddle格式模型的参数名字
  qat_image_path: ./COCO_train_320            #量化蒸馏训练使用的数据集,此例为少量无标签数据, 选自COCO2017训练集中的前320张图片, 做少量数据训练
  ptq_image_path: ./COCO_val_320              #离线训练使用的Carlibration数据集, 选自COCO2017验证集中的前320张图片.
  input_list: ['image','scale_factor']        #待量化的模型的输入名字
  qat_preprocess: ppyoloe_plus_withNMS_image_preprocess #模型量化蒸馏训练时,对数据做的预处理函数, 用户可以在 ../fdquant/dataset.py 中修改或自行编写新的预处理函数, 来支自定义模型的量化
  ptq_preprocess: ppyoloe_plus_withNMS_image_preprocess #模型离线量化时,对数据做的预处理函数, 用户可以在 ../fdquant/dataset.py 中修改或自行编写新的预处理函数, 来支自定义模型的量化
  qat_batch_size: 4                           #量化蒸馏训练时的batch_size, 若为onnx格式的模型,此处只能为1


#量化蒸馏训练配置
Distillation:
  alpha: 1.0                                  #蒸馏loss所占权重
  loss: soft_label                            #蒸馏loss算法

QuantAware:
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

FastDeploy一键压缩功能由PaddeSlim助力, 更详细的量化配置方法请参考:
[自动化压缩超参详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/hyperparameter_tutorial.md)
