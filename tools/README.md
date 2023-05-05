# FastDeploy 工具包
FastDeploy提供了一系列高效易用的工具优化部署体验, 提升推理性能.

- [1.自动压缩工具包](#1)
- [2.模型转换工具包](#2)
- [3.paddle2coreml工具包](#3)

<p id="1"></p>

## 一键模型自动化压缩工具

FastDeploy基于PaddleSlim的Auto Compression Toolkit(ACT), 给用户提供了一键模型自动化压缩的工具, 用户可以轻松地通过一行命令对模型进行自动化压缩, 并在FastDeploy上部署压缩后的模型, 提升推理速度. 本文档将以FastDeploy一键模型自动化压缩工具为例, 介绍如何安装此工具, 并提供相应的使用文档.

### 环境准备
1.用户参考PaddlePaddle官网, 安装Paddle 2.4 版本
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.安装PaddleSlim 2.4 版本
```bash
pip install paddleslim==2.4.0
```

3.安装fastdeploy-tools工具包
```bash
# 通过pip安装fastdeploy-tools. 此工具包目前支持模型一键自动化压缩和模型转换的功能.
# FastDeploy的python包已包含此工具, 不需重复安装.
pip install fastdeploy-tools==0.0.1
```

### 一键模型自动化压缩工具的使用
按照以上步骤成功安装后,即可使用FastDeploy一键模型自动化压缩工具, 示例如下.

```bash
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```
详细使用文档请参考[FastDeploy一键模型自动化压缩工具](./common_tools/auto_compression/README.md)

<p id="2"></p>

## 模型转换工具

FastDeploy 基于 X2Paddle 为用户提供了模型转换的工具, 用户可以轻松地通过一行命令将外部框架模型快速迁移至飞桨框架，目前支持 ONNX、TensorFlow 以及 Caffe，支持大部分主流的CV和NLP的模型转换。

### 环境准备

1. PaddlePaddle 安装，可参考如下文档快速安装
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2. X2Paddle 安装

如需使用稳定版本，可通过pip方式安装X2Paddle：
```shell
pip install x2paddle
```

如需体验最新功能，可使用源码安装方式：
```shell
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
python setup.py install
```

### 使用方式

按照以上步骤成功安装后,即可使用 FastDeploy 一键转换工具, 示例如下:

```bash
fastdeploy convert --framework onnx --model yolov5s.onnx --save_dir pd_model
```

更多详细内容可参考[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)

## paddle2coreml工具

FastDeploy 基于 paddle2coreml 为用户提供了模型转换的工具, 用户可以轻松地通过一行命令将飞桨模型快速迁移至苹果电脑和手机端。

### 环境准备

1. PaddlePaddle 安装，可参考如下文档快速安装
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```
2. paddle2coreml 安装

可通过pip方式安装paddle2coreml：
```shell
pip install paddle2coreml
```
3. 使用方式

按照以上步骤成功安装后,即可使用 FastDeploy paddle2coreml 一键转换工具, 示例如下:

```bash
fastdeploy paddle2coreml --p2c_paddle_model_dir path/to/paddle_model --p2c_coreml_model_dir path/to/coreml_model --p2c_input_names "input1 input2" --p2c_input_shapes "1,3,224,224 1,4,64,64" --p2c_input_dtypes "float32 int32" --p2c_output_names "output1 output2" 
```
注意，--p2c_input_names 与 --p2c_output_names 两个参数须与paddle模型的输入输出名字一致。

