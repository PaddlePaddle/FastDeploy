# 简介

本文档以[千分类模型_MobileNetV3](https://ai.baidu.com/easyedge/app/openSource)为例，介绍FastDeploy中的模型SDK， 在**ARM Linux Python** 环境下： （1)**服务化**推理部署步骤； （2）介绍模型推流全流程API，方便开发者了解项目后二次开发。其中ARM Linux Python请参考[ARM Linux C++环境下的HTTP推理部署](./arm_linux_cpp_sdk_serving.md)文档。

**注意**：部分模型（如OCR等）不支持服务化推理。

<!--ts-->

* [简介](#简介)

* [环境准备](#环境准备)
  
  * [1.SDK下载](#1sdk下载)
  * [2.硬件支持](#2硬件支持)
  * [3.Python环境](#3python环境)
  * [4.安装依赖](#4安装依赖)
    * [4.1.安装paddlepaddle](#41安装paddlepaddle)
    * [4.2.安装EasyEdge Python Wheel 包](#42安装easyedge-python-wheel-包)

* [快速开始](#快速开始)
  
  * [1.文件结构说明](#1文件结构说明)
  * [2.测试Serving服务](#2测试serving服务)
    * [2.1 启动HTTP预测服务](#21-启动http预测服务)

* [HTTP API流程详解](#http-api流程详解)
  
  * [1. 开启http服务](#1-开启http服务)
  * [2. 请求http服务](#2-请求http服务)
    * [2.1 http 请求方式：不使用图片base64格式](#21-http-请求方式不使用图片base64格式)
  * [3. http返回数据](#3-http返回数据)

* [FAQ](#faq)
  
  <!--te-->

# 环境准备

## 1.SDK下载

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GitHub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压缩后的文件结构如下。

```shell
EasyEdge-Linux-x86-[部署芯片]
├── RES      # 模型文件资源文件夹，可替换为其他模型
├── README.md
├── cpp     # C++ SDK
└── python  # Python SDK
```

## 2.硬件支持

目前支持的ARM架构：aarch64 、armv7hf

## 3.Python环境

> ARM Linux SDK仅支持Python 3.6

使用如下命令获取已安装的Python版本号。如果本机的版本不匹配，需要根据ARM Linux下Python安装方式进行安装。（不建议在ARM Linux下使用conda，因为ARM Linux场景通常资源很有限）

```shell
$python3 --version
```

接着使用如下命令确认pip的版本是否满足要求，要求pip版本为20.2.2或更高版本。详细的pip安装过程可以参考[官网教程](https://pip.pypa.io/en/stable/installation/)。

```shell
$python3 -m pip --version
```

## 4.安装依赖

### 4.1.安装paddlepaddle

根据具体的部署芯片（CPU/GPU）安装对应的PaddlePaddle的whl包。

`armv8 CPU平台`可以使用如下命令进行安装：

```shell
python3 -m pip install http://aipe-easyedge-public.bj.bcebos.com/easydeploy/paddlelite-2.11-cp36-cp36m-linux_aarch64.whl 
```

### 4.2.安装EasyEdge Python Wheel 包

在`python`目录下，安装特定Python版本的EasyEdge Wheel包。`armv8 CPU平台`可以使用如下命令进行安装：

```shell
python3 -m pip install -U BaiduAI_EasyEdge_SDK-1.3.1-cp36-cp36m-linux_aarch64.whl
```

# 二.快速开始

## 1.文件结构说明

Python SDK文件结构如下：

```shell
EasyEdge-Linux-x86--[部署芯片]
├──...
├──python            # Linux Python SDK
    ├──              # 特定Python版本的EasyEdge Wheel包, 二次开发可使用
    ├── BBaiduAI_EasyEdge_SDK-1.3.1-cp36-cp36m-linux_aarch64.whl
    ├── infer_demo   # demo体验完整文件
    │   ├──  demo_xxx.py     # 包含前后处理的端到端推理demo文件
    │   └──  demo_serving.py # 提供http服务的demo文件
    ├── tensor_demo  # 学习自定义算法前后处理时使用
    │   └──  demo_xxx.py
```

## 2.测试Serving服务

> 模型资源文件默认已经打包在开发者下载的SDK包中， 默认为`RES`目录。

### 2.1 启动HTTP预测服务

指定对应的模型文件夹（默认为`RES`）、设备ip和指定端口号，运行如下命令。

```shell
python3 demo_serving.py {模型RES文件夹} {host, default 0.0.0.0} {port, default 24401}
```

成功启动后，终端中会显示如下字样。

```shell
...
* Running on {host ip}:24401
```

如果是在局域网内的机器上部署，开发者此时可以打开浏览器，输入`http://{host ip}:24401`，选择图片来进行测试，运行效果如下。

<img src="https://user-images.githubusercontent.com/54695910/175854073-fb8189e5-0ffb-472c-a17d-0f35aa6a8418.png" style="zoom:50%;" />

如果是在远程机器上部署，那么可以参考`demo_serving.py`中的 `http_client_test()函数`请求http服务来执行推理。

# 三. HTTP API流程详解

## 1. 开启http服务

http服务的启动使用`demo_serving.py`文件

```python
class Serving(object):
    """
        SDK local serving
    """

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
      """
          Args:
              host : str
              port : str
              device : BaiduAI.EasyEdge.Device，比如：Device.CPU
              engine : BaiduAI.EasyEdge.Engine， 比如： Engine.PADDLE_FLUID
      """
        self.run_serving_with_flask(host, port, device, engine, service_id, device_id, **kwargs)
```

## 2. 请求http服务

> 开发者可以打开浏览器，`http://{设备ip}:24401`，选择图片来进行测试。

### 2.1 http 请求方式：不使用图片base64格式

URL中的get参数：

| 参数        | 说明        | 默认值              |
| --------- | --------- | ---------------- |
| threshold | 阈值过滤， 0~1 | 如不提供，则会使用模型的推荐阈值 |

HTTP POST Body即为图片的二进制内容

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

*** 关于图像分割mask ***

```
cv::Mat mask为图像掩码的二维数组
{
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
}
其中1代表为目标区域，0代表非目标区域
```

# FAQ

1.执行infer_demo文件时，提示your generated code is out of date and must be regenerated with protoc >= 3.19.0

    进入当前项目，首先卸载protobuf
    
    ```shell
    python3 -m pip uninstall protobuf
    ```
    
    安装低版本protobuf
    
    ```shell
    python3 -m pip install protobuf==3.19.0
    ```
