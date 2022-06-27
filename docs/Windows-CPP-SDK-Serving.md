# 简介

本文档以[千分类模型_MobileNetV3](https://ai.baidu.com/easyedge/app/openSource)为例，本文档介绍FastDeploy中的模型SDK，在**Intel x86_64 / NVIDIA GPU、Windows操作系统** 的C++环境：（1）HTTP服务化推理部署步骤，（2）介绍推理全流程API，方便开发者了解项目后二次开发。
如果开发者对Python语言的相关能力感兴趣，可以参考Windows Python请参考[Windows Python环境下的推理部署](./Windows-Python-SDK-Serving.md)文档。

<!--ts-->

* [简介](#简介)

* [环境准备](#环境准备)
  
  * [1. SDK下载](#1-sdk下载)
  * [2. CPP环境](#2-cpp环境)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. 测试EasyEdge服务](#2-测试easyedge服务)
  * [3. 启动HTTP预测服务](#3-启动http预测服务)
  * [4. 编译Demo](#4-编译demo)

* [HTTP API流程详解](#http-api流程详解)
  
  * [1. 开启http服务](#1-开启http服务)
  * [2. 请求http服务](#2-请求http服务)
    * [2.1 http 请求方式一:不使用图片base64格式](#21-http-请求方式一不使用图片base64格式)
    * [2.2 http 请求方法二:使用图片base64格式](#22-http-请求方法二使用图片base64格式)
  * [3. http 返回数据](#3-http-返回数据)

* [FAQ](#faq)
  
  <!--te-->

# 环境准备

## 1. SDK下载

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压缩后的文件结构如`快速开始`中[1项目介绍说明](#1-%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84%E8%AF%B4%E6%98%8E)介绍。

```shell

```

## 2. CPP环境

> 建议使用Microsoft Visual Studio 2015及以上版本，获取核心 C 和 C++ 支持，安装时请选择“使用 C++ 的桌面开发”工作负载。

# 快速开始

## 1. 项目结构说明

```shell
EasyEdge-win-xxx
        ├── data
        │   ├── model    # 模型文件资源文件夹，可替换为其他模型
        │   └── config   # 配置文件
        ├── bin          # demo二进制程序
           │   ├── xxx_image    # 预测图像demo
           │   ├── xxx_video    # 预测视频流demo
        │  └── xxx_serving  # 启动http预测服务demo
        ├── dll          # demo二进制程序依赖的动态库
        ├── ...          # 二次开发依赖的文件
        ├── python       # Python SDK文件
        ├── EasyEdge.exe # EasyEdge服务
        └── README.md    # 环境说明
```

## 2. 测试EasyEdge服务

> 模型资源文件默认已经打包在开发者下载的SDK包中，请先将zip包整体拷贝到具体运行的设备中，再解压缩使用。

SDK下载完成后，双击打开EasyEdge.exe启动推理服务，输入要绑定的Host ip及端口号Port，点击启动服务。

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854086-d507c288-56c8-4fa9-a00c-9d3cfeaac1c8.png" alt="图片" style="zoom: 67%;" />
</div>

服务启动后，打开浏览器输入`http://{Host ip}:{Port}`，添加图片或者视频来进行测试。

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/175854073-fb8189e5-0ffb-472c-a17d-0f35aa6a8418.png" style="zoom:67%;" />
</div>
## 3. 启动HTTP预测服务

除了通过上述方式外，您还可以使用bin目录下的可执行文件来体验单一的功能。在dll目录下，点击右键，选择"在终端打开"，执行如下命令。

> 需要将bin目录下的可执行文件移动到dll目录下执行，或者将dll目录添加到系统环境变量中。

```
.\easyedge_serving {模型model文件夹路径} 
```

启动后，日志中会显示如下字样。

```
HTTP is now serving at 0.0.0.0:24401
```

此时，开发者可以打开浏览器，`http://127.0.0.1:24401`，执行和之前一样的操作即可。

![](https://user-images.githubusercontent.com/54695910/175854073-fb8189e5-0ffb-472c-a17d-0f35aa6a8418.png)

## 4. 编译Demo

在[项目结构说明](#1项目结构说明)中，`bin`路径下的可执行文件是由`src`下的对应文件编译得到的，具体的编译命令如下。

```
cd src
mkdir build && cd build
cmake .. && make
```

编译完成后，在build文件夹下会生成编译好的可执行文件，如图像推理的二进制文件：`build/demo_serving/easyedge_serving`。

# HTTP API流程详解

本章节主要结合[2.1 HTTP Demo](#4)的API介绍，方便开发者学习并将运行库嵌入到开发者的程序当中，更详细的API请参考`include/easyedge/easyedge*.h`文件。http服务包含服务端和客户端，目前支持的能力包括以下几种方式，Demo中提供了不使用图片base格式的`方式一：浏览器请求的方式`，其他几种方式开发者根据个人需要，选择开发。

## 1. 开启http服务

http服务的启动可直接使用`bin/easyedge_serving`，或参考`src/demo_serving.cpp`文件修改相关逻辑

```cpp
 /**
     * @brief 开启一个简单的demo http服务。
     * 该方法会block直到收到sigint/sigterm。
     * http服务里，图片的解码运行在cpu之上，可能会降低推理速度。
     * @tparam ConfigT
     * @param config
     * @param host
     * @param port
     * @param service_id service_id  user parameter, uri '/get/service_id' will respond this value with 'text/plain'
     * @param instance_num 实例数量，根据内存/显存/时延要求调整
     * @return
     */
    template<typename ConfigT>
    int start_http_server(
            const ConfigT &config,
            const std::string &host,
            int port,
            const std::string &service_id,
            int instance_num = 1);
```

## 2. 请求http服务

> 开发者可以打开浏览器，`http://{设备ip}:24401`，选择图片来进行测试。

### 2.1 http 请求方式一:不使用图片base64格式

URL中的get参数：

| 参数        | 说明        | 默认值              |
| --------- | --------- | ---------------- |
| threshold | 阈值过滤， 0~1 | 如不提供，则会使用模型的推荐阈值 |

HTTP POST Body即为图片的二进制内容(无需base64, 无需json)

Python请求示例

```Python
import requests

with open('./1.jpg', 'rb') as f:
    img = f.read()
    result = requests.post(
        'http://127.0.0.1:24401/',
        params={'threshold': 0.1},
        data=img).json()
```

### 2.2 http 请求方法二:使用图片base64格式

HTTP方法：POST
Header如下：

| 参数           | 值                |
| ------------ | ---------------- |
| Content-Type | application/json |

**Body请求填写**：

- 分类网络：
  body 中请求示例
  
  ```
  {
    "image": "<base64数据>"
    "top_num": 5
  }
  ```
  
  body中参数详情

| 参数      | 是否必选 | 类型     | 可选值范围 | 说明                                                                                  |
| ------- | ---- | ------ | ----- | ----------------------------------------------------------------------------------- |
| image   | 是    | string | -     | 图像数据，base64编码，要求base64图片编码后大小不超过4M,最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式 **注意去掉头部** |
| top_num | 否    | number | -     | 返回分类数量，不填该参数，则默认返回全部分类结果                                                            |

- 检测和分割网络：
  Body请求示例：
  
  ```
  {
    "image": "<base64数据>"
  }
  ```
  
  body中参数详情：

| 参数        | 是否必选 | 类型     | 可选值范围 | 说明                                                                                  |
| --------- | ---- | ------ | ----- | ----------------------------------------------------------------------------------- |
| image     | 是    | string | -     | 图像数据，base64编码，要求base64图片编码后大小不超过4M,最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式 **注意去掉头部** |
| threshold | 否    | number | -     | 默认为推荐阈值，也可自行根据需要进行设置                                                                |

## 3. http 返回数据

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

*** 关于矩形坐标 ***

x1 * 图片宽度 = 检测框的左上角的横坐标

y1 * 图片高度 = 检测框的左上角的纵坐标

x2 * 图片宽度 = 检测框的右下角的横坐标

y2 * 图片高度 = 检测框的右下角的纵坐标

*** 关于分割模型 ***

其中，mask为分割模型的游程编码，解析方式可参考 [http demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)。

# FAQ

1. 执行infer_demo文件时，提示your generated code is out of date and must be regenerated with protoc >= 3.19.0

进入当前项目，首先卸载protobuf

```shell
python3 -m pip uninstall protobuf
```

安装低版本protobuf

```shell
python3 -m pip install protobuf==3.19.0
```
