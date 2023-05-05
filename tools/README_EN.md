# FastDeploy Toolkit
FastDeploy provides a series of efficient and easy-to-use tools to optimize the deployment experience and improve inference performance.

- [1.Auto Compression Tool](#1)
- [2.Model Conversion Tool](#2)

<p id="1"></p>

## One-Click Model Auto Compression Tool

Based on PaddleSlim's Auto Compression Toolkit (ACT), FastDeploy provides users with a one-click model automation compression tool that allows users to easily compress the model with a single command. This document will take FastDeploy's one-click model automation compression tool as an example, introduce how to install the tool, and provide the corresponding documentation for usage.

### Environmental Preparation
1.Install PaddlePaddle 2.4 version
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.Install PaddleSlim 2.4 version
```bash
pip install paddleslim==2.4.0
```

3.Install fastdeploy-tools package
```bash
# Installing fastdeploy-tools via pip
# This tool is included in the python installer of FastDeploy, so you don't need to install it again.
pip install fastdeploy-tools==0.0.1

```

### The Usage of One-Click Model Auto Compression Tool
After the above steps are successfully installed, you can use FastDeploy one-click model automation compression tool, as shown in the following example.
```bash
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```
For detailed documentation, please refer to [FastDeploy One-Click Model Auto Compression Tool](./common_tools/auto_compression/README_EN.md)

<p id="2"></p>

## Model Conversion Tool

Based on X2Paddle, FastDeploy provides users with a model conversion tool. Users can easily migrate external framework models to the Paddle framework with one line of commands. Currently, ONNX, TensorFlow and Caffe are supported, and most mainstream CV and NLP model conversions are supported.

### Environmental Preparation

1. Install PaddlePaddle, refer to the following documents for quick installation
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2. Install X2Paddle

To use the stable version, install X2Paddle via pip:
```shell
pip install x2paddle
```

To experience the latest features, you can use the source installation method:
```shell
git clone https://github.com/PaddlePaddle/X2Paddle.git
cd X2Paddle
python setup.py install
```

### How to use

After successful installation according to the above steps, you can use the FastDeploy one-click conversion tool. The example is as follows:

```bash
fastdeploy convert --framework onnx --model yolov5s.onnx --save_dir pd_model
```

For more details, please refer to[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)

## paddle2coreml tool

FastDeploy provides users with a model conversion tool based on paddle2coreml, which allows users to easily migrate PaddlePaddle models to Apple computers and mobile devices with a single command.

### Environment Preparation

1. PaddlePaddle installation, please refer to the following document for quick installation:
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```
2. paddle2coreml installation

paddle2coreml can be installed using pip:
```shell
pip install paddle2coreml
```
3. Usage

After successfully installing as described above, you can use the FastDeploy paddle2coreml one-click conversion tool, as shown below:

```bash
fastdeploy paddle2coreml --p2c_paddle_model_dir path/to/paddle_model --p2c_coreml_model_dir path/to/coreml_model --p2c_input_names "input1 input2" --p2c_input_shapes "1,3,224,224 1,4,64,64" --p2c_input_dtypes "float32 int32" --p2c_output_names "output1 output2" 
```