# StableDiffusion 服务化部署文档

本文档描述StableDiffusion模型（以下简称SD）服务化部署流程:

**注意**:目前该版本的SD模型仅保证可以正确运行在NVIDIA-A系列显卡中，其他NVIDIA系列显卡不保证能正确加载模型和预测。

## 一.环境安装与模型下载

本文中所有的代码均在models目录下，其结构如下:

```
models
`|-- env_bash.sh
 |-- export_model.py
 |-- stable_diffusion
 `-- client.py
```

- `env_bash.sh`为环境安装和模型下载脚本文件。
- `export_model.py`为下载SD模型的脚本代码，会被`env_bash.sh`调用来下载模型并保存到Server端目录下。
- `stable_diffusion`为服务端相关代码。
- `client.py`为客户端预测代码。

首先，拉取并创建docker，将models路径投影到docker中，并执行脚本安装依赖和下载模型。
```
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-gpu-cuda11.4-trt8.5-21.10
nvidia-docker run --name sd_serving -dit --net=host -v $PWD:/models registry.baidubce.com/paddlepaddle/fastdeploy:1.0.4-gpu-cuda11.4-trt8.5-21.10 bash
docker exec -it -u root sd_serving bash
cd /models
bash env_bash.sh
```
执行bash env_bash.sh会安装相关的python包依赖和下载SD模型，正确下载的模型目录结构如下:

```shell
stable-diffusion-v1-5/
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── tokenizer
│   ├── tokenizer_config.json
│   ├── merges.txt
│   ├── vocab.json
│   └── special_tokens_map.json
├── text_encoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── unet
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── vae_decoder
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── vae_encoder
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

**注意**:如果运行`env_bash.sh`时发生错误，请将/opt/fastdeploy/third_libs/install/xxxxx/libdnnl.so.2 替换为paddle-develop版本编译的最新版`libdnnl.so.2`。

## 二.Server端

Server端代码均在models/stable_diffusion/目录下，其结构如下:

```
models
`-- stable_diffusion
    |-- 1
    |   |-- model.py
    |   |-- stable-diffusion-v1-5
    `-- config.pbtxt
```

- `config.pbtxt`为Triton服务的接口（输入输出）和配置选项。
- `model.py`为Triton服务层代码文件。
- `stable-diffusion-v1-5`为上述步骤下载的SD模型。

## 三.启动Triton服务

在当前目录下（models目录中）运行以下命令行
```
fastdeployserver --model-repository=.
```

参数:
  - `model-repository`(required): 服务端代码存放的文件夹路径.
  - `http-port`(optional): HTTP服务的端口号. 默认: `8000`. 本示例中未使用该端口.
  - `grpc-port`(optional): GRPC服务的端口号. 默认: `8001`.
  - `metrics-port`(optional): 服务端指标的端口号. 默认: `8002`. 本示例中未使用该端口.

**注意**:首次加载会比较慢，用户可根据自身需求修改`model.py`中的相关参数和配置。

服务的调用输入为String类型的一段文字，例如:"a photo of an astronaut riding a horse on mars"。

服务的调用输出为生成的图片的base64编码的String。

具体接口配置文件在`config.pbtxt`中，定义如下：
```
input [
  {
    name: "INPUT_0"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

output [
  {
    name: "OUTPUT_0"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
```

## 四.Client端发起预测

在当前目录下（models目录中）运行以下命令行

```
python client.py
```

**注意**:`client.py`仅为示例代码，用户可根据自身需求参考`client.py`进行修改。
