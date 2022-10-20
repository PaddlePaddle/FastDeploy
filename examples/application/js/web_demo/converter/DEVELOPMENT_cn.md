[English](./README.md)
# paddlejs-converter

paddlejs-converter 是适用于 Paddle.js 的模型转换工具，其作用是将 PaddlePaddle 模型（或称为 fluid 模型）转化为浏览器友好的格式，以供Paddle.js在浏览器等环境中加载预测使用。此外，paddlejs-converter 还提供了强大的模型优化能力，帮助开发者对模型结构进行优化，提高运行时性能。

## 1. 使用教程

### 1.1. 环境搭建
#### Python 版本确认
确认运行平台的 Python 环境与版本是否满足要求，若使用 Python3 ，则可能需要将后续命令中的 `python` 换成 `python3`：
- Python3： 3.5.1+ / 3.6 / 3.7
- Python2： 2.7.15+

#### 安装虚拟环境
*由于开发环境可能安装了多个版本的 Python，相关依赖包可能存在不同的版本，为避免产生冲突，**强烈建议**使用 Python 虚拟环境执行转换工具所需的各项命令，以免产生各种问题。若不使用虚拟环境或已安装虚拟环境，可跳过该步骤。*

以 Anaconda 为例：
前往 [Anaconda](https://www.anaconda.com/) 主页，选择对应平台、Python 版本的 Anaconda 按照官方提示，进行安装；

安装完毕后，在命令行执行以下命令，创建Python 虚拟环境：
``` bash
conda create --name <your_env_name>
```

执行以下命令，切换至虚拟环境
``` bash
# Linux 或 macOS下请执行
source activate <your_env_name>

# Windows 下请执行
activate <your_env_name>
```

#### 安装依赖
- 如果`不需要`使用优化模型的能力，执行命令：
``` bash
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
- 如果`需要`使用优化模型的能力，执行命令：
``` bash
python -m pip install paddlepaddle paddlelite==2.6.0 -i https://mirror.baidu.com/pypi/simple
```

### 1.2. 快速上手
- 如果待转换的 fluid 模型为`合并参数文件`，即一个模型对应一个参数文件：
``` bash
python convertToPaddleJSModel.py --modelPath=<fluid_model_file_path> --paramPath=<fluid_param_file_path> --outputDir=<paddlejs_model_directory>
```
- 如果待转换的 fluid 模型为`分片参数文件`，即一个模型文件对应多个参数文件：
``` bash
# 注意，使用这种方式调用转换器，需要保证 inputDir 中，模型文件名为'__model__'
python convertToPaddleJSModel.py --inputDir=<fluid_model_directory> --outputDir=<paddlejs_model_directory>
````
模型转换器将生成以下两种类型的文件以供 Paddle.js 使用：

- model.json (模型结构与参数清单)
- chunk_\*.dat (二进制参数文件集合)

## 2. 详细文档
参数 |  描述
:-: | :-:
--inputDir | fluid 模型所在目录，当且仅当使用分片参数文件时使用该参数，将忽略 `modelPath` 和 `paramPath` 参数，且模型文件名必须为`__model__`
--modelPath | fluid 模型文件所在路径，使用合并参数文件时使用该参数
--paramPath | fluid 参数文件所在路径，使用合并参数文件时使用该参数
--outputDir | `必要参数`， Paddle.js 模型输出路径
--disableOptimize | 是否关闭模型优化， `1` 为关闭优化，`0` 为开启优化（需安装 PaddleLite ），默认执行优化
--logModelInfo | 是否打印模型结构信息， `0` 为不打印， `1` 为打印，默认不打印
--sliceDataSize | 分片输出 Paddle.js 参数文件时，每片文件的大小，单位：KB，默认 4096
--useGPUOpt | 是否开启模型 GPU 优化，默认不开启（当模型准备运行在 webgl/webgpu 计算方案时，可以设置为 True 开启，在 wasm/plainjs 方案，则不用开启）

## 3. 其他信息
若需要转换的模型为 `TensorFlow/Caffe/ONNX` 格式，可使用 PaddlePaddle 项目下的 `X2Paddle`工具，将其他格式的模型转为 fluid 模型后，再使用本工具转化为 Paddle.js 模型。
详细请参考 [X2Paddle 项目](https://github.com/PaddlePaddle/X2Paddle)