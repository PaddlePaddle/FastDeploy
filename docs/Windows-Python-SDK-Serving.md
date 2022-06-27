# 简介

本文档以[千分类模型_MobileNetV3](https://ai.baidu.com/easyedge/app/openSource)为例，介绍FastDeploy中的模型SDK，在**Intel x86_64 /NVIDIA GPU、Windows操作系统** 的Python环境：（1）HTTP服务化推理部署步骤，（2）介绍推理全流程API，方便开发者了解项目后二次开发。
如果开发者对C++语言的相关能力感兴趣，可以参考Windows C++请参考[Windows C++环境下的推理部署](./Windows-CPP-SDK-Serving.md)文档。

<!--ts-->

* [简介](#简介)

* [环境准备](#环境准备)
  
  * [1. SDK下载](#1-sdk下载)
  * [2. Python环境](#2-python环境)
  * [3. 安装依赖](#3-安装依赖)
    * [3.1 安装paddlepaddle](#31-安装paddlepaddle)
    * [3.2 安装EasyEdge Python Wheel 包](#32-安装easyedge-python-wheel-包)

* [快速开始](#快速开始)
  
  * [1. 文件结构说明](#1-文件结构说明)
  * [2. 测试Demo](#2-测试demo)
    * [2.1 启动HTTP预测服务](#21-启动http预测服务)

* [HTTP API流程详解](#http-api流程详解)
  
  * [1. 开启http服务](#1-开启http服务)
  
  * [2. 请求http服务](#2-请求http服务)
    
    * [2.1 http 请求方式:不使用图片base64格式](#21-http-请求方式不使用图片base64格式)
  
  * [3. http返回数据](#3-http返回数据)
    
    <!--te-->

# 环境准备

## 1. SDK下载

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压缩后的文件结构如下所示：

```shell
EasyEdge-win-[部署芯片]
├── data          # 模型文件资源文件夹，可替换为其他模型
├── ...           # C++/C# 相关文件
├── python        # Python SDK文件
├── EasyEdge.exe  # 主程序
└── README.md     # 环境说明
```

## 2. Python环境

> 当前SDK仅支持Python 3.7

打开命令行工具，使用如下命令获取已安装的Python版本号。如果还没有安装Python环境，可以前往[官网](https://www.python.org/)下载Python 3.7对应的安装程序，特别要注意勾上`Add Python 3.7 to PATH`，然后点“Install Now”即可完成安装。

```shell
python --version
```

如果本机的版本不匹配，建议使用[pyenv](https://github.com/pyenv/pyenv)、[anaconda](https://www.anaconda.com/)等Python版本管理工具对Python SDK所在目录进行配置。

接着使用如下命令确认pip的版本是否满足要求，要求pip版本为20.2.2或更高版本。详细的pip安装过程可以参考[官网教程](https://pip.pypa.io/en/stable/installation/)。

```shell
python -m pip --version
```

## 3. 安装依赖

### 3.1 安装paddlepaddle

根据具体的部署芯片（CPU/GPU）安装对应的PaddlePaddle的whl包。`x86_64 CPU` 平台可以使用如下命令进行安装：

```shell
python -m pip install paddlepaddle==2.2.2 -i https://mirror.baidu.com/pypi/simple 
```

`NVIDIA GPU平台`的详细安装教程可以参考[官网Paddle安装教程](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。  

> 使用 NVIDIA GPU 预测时，必须满足：  
> 
>     1. 机器已安装 cuda, cudnn  
> 
> 2. 已正确安装对应 cuda 版本的paddle 版本  
> 3. 通过设置环境变量`FLAGS_fraction_of_gpu_memory_to_use`设置合理的初始内存使用比例

### 3.2 安装EasyEdge Python Wheel 包

在`python`目录下，安装Python3.7版本对应的EasyEdge Wheel包。对`x86_64 CPU` 或 `x86_64 Nvidia GPU平台 `可以使用如下命令进行安装，具体名称以 Python SDK 包中的 whl 为准。

```shell
python -m pip install -U BaiduAI_EasyEdge_SDK-{SDK版本号}-cp37-cp37m-win_amd64.whl
```

# 快速开始

## 1. 文件结构说明

Python SDK文件结构如下：

```shell
EasyEdge-win-[部署芯片]
├── data                  # 模型文件资源文件夹，可替换为其他模型
│   ├── model             # 模型文件资源文件夹，可替换为其他模型
│   └── config            # 配置文件
├── ...                   # C++/C# 相关文件
├── python                # Python SDK文件
│   ├── # 特定Python 3.7版本的EasyEdge Wheel包, 二次开发可使用
│   ├── BaiduAI_EasyEdge_SDK-${SDK版本号}-cp37-cp37m-win_amd64.whl
│   ├── requirements.txt  # 
│   ├── infer_demo        # demo体验完整文件
│   │   ├──  demo_xxx.py  # 包含前后处理的端到端推理demo文件
│   │   └──  demo_serving.py # 提供http服务的demo文件
│   └── tensor_demo       # tensor in/out demo文件
```

## 2. 测试Demo

### 2.1 启动HTTP预测服务

```shell
python demo_serving.py {模型model文件夹} {host, default 0.0.0.0} {port, default 24401}
```

成功启动后，终端中会显示如下字样。

```shell
2022-06-14 18:45:15 INFO [EasyEdge] [demo_serving.py:50] 21212: Init paddlefluid engine...
2022-06-14 18:45:16 INFO [EasyEdge] [demo_serving.py:50] 21212: Paddle version: 2.2.2
 * Serving Flask app 'Serving' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:24401
 * Running on http://192.168.3.17:24401 (Press CTRL+C to quit)
```

开发者此时可以打开浏览器，输入`http://{host ip}:24401`，选择图片或者视频来进行测试，运行效果如下。

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854073-fb8189e5-0ffb-472c-a17d-0f35aa6a8418.png" style="zoom:50%;" />
</div>

# HTTP API流程详解

本章节主要结合前文的Demo示例来对API进行介绍，方便开发者学习并将运行库嵌入到开发者的程序当中，更详细的API请参考对应的Python文件。http服务包含服务端和客户端，Demo中提供了不使用图片base格式的`方式一：浏览器请求的方式`，其他几种方式开发者根据个人需要，选择开发。

## 1. 开启http服务

http服务的启动使用`demo_serving.py`文件

```python
class Serving(object):
    """        SDK local serving    """

    def __init__(self, model_dir, license='', model_filename='model', params_filename='params'):

        self.program = None
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.params_filename = params_filename
        self.program_lock = threading.Lock()
        self.license_key = license
        # 只有ObjectTracking会初始化video_processor
        self.video_processor = None

     def run(self, host, port, device, engine=Engine.PADDLE_FLUID, service_id=0, device_id=0, **kwargs):
      """          Args:              host : str              port : str              device : BaiduAI.EasyEdge.Device，比如：Device.CPU              engine : BaiduAI.EasyEdge.Engine， 比如： Engine.PADDLE_FLUID      """
        self.run_serving_with_flask(host, port, device, engine, service_id, device_id, **kwargs)
```

## 2. 请求http服务

> 开发者可以打开浏览器，`http://{设备ip}:24401`，选择图片来进行测试。

### 2.1 http 请求方式:不使用图片base64格式

URL中的get参数：

| 参数        | 说明        | 默认值              |
| --------- | --------- | ---------------- |
| threshold | 阈值过滤， 0~1 | 如不提供，则会使用模型的推荐阈值 |

HTTP POST Body即为图片的二进制内容。

Python请求示例

```python
import requests

with open('./1.jpg', 'rb') as f:
    img = f.read()
    result = requests.post(
        'http://127.0.0.1:24401/',
        params={'threshold': 0.1},
        data=img).json()
```

## 3. http返回数据

| 字段         | 类型说明   | 其他                                   |
| ---------- | ------ | ------------------------------------ |
| error_code | Number | 0为成功,非0参考message获得具体错误信息             |
| results    | Array  | 内容为具体的识别结果。其中字段的具体含义请参考`预测图像-返回格式`一节 |
| cost_ms    | Number | 预测耗时ms，不含网络交互时间                      |

返回示例

```json
{
    "cost_ms": 52,
    "error_code": 0,
    "results": [
        {
            "confidence": 0.94482421875,
            "index": 1,
            "label": "IronMan",
            "x1": 0.059185408055782318,
            "x2": 0.18795496225357056,
            "y1": 0.14762254059314728,
            "y2": 0.52510076761245728,
            "mask": "...",  // 图像分割模型字段
            "trackId": 0,  // 目标追踪模型字段
        },

      ]
}
```

***关于矩形坐标***

x1 * 图片宽度 = 检测框的左上角的横坐标

y1 * 图片高度 = 检测框的左上角的纵坐标

x2 * 图片宽度 = 检测框的右下角的横坐标

y2 * 图片高度 = 检测框的右下角的纵坐标

***关于分割模型***

其中，mask为分割模型的游程编码，解析方式可参考 [demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)。

**FAQ**

1. 执行infer_demo文件时，提示your generated code is out of date and must be regenerated with protoc >= 3.19.0

进入当前项目，首先卸载protobuf

```shell
python3 -m pip uninstall protobuf
```

安装低版本protobuf

```shell
python3 -m pip install protobuf==3.19.0
```
