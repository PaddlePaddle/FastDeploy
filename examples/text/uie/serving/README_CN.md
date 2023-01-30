[English](README.md) | 简体中文

# UIE 服务化部署示例

在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](../../../../serving/README_CN.md)

## 准备模型

下载UIE-Base模型(如果有已训练好的模型，跳过此步骤):
```bash
# 下载UIE模型文件和词表，以uie-base模型为例
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz

# 将下载的模型移动到模型仓库目录
mv uie-base/inference.pdmodel models/uie/1/model.pdmodel
mv uie-base/inference.pdiparams models/uie/1/model.pdiparams
```

模型下载移动好之后，目录结构如下:
```
models
└── uie
    ├── 1
    │   ├── model.pdiparams
    │   ├── model.pdmodel
    │   ├── model.py
    │   └── vocab.txt
    └── config.pbtxt
```

## 拉取并运行镜像
```bash
# x.y.z为镜像版本号，需参照serving文档替换为数字
# GPU镜像
docker pull paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU镜像
docker pull paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /uie_serving 目录
docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v `pwd`/:/uie_serving paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash

# 启动服务(不设置CUDA_VISIBLE_DEVICES环境变量，会拥有所有GPU卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/uie_serving/models --backend-config=python,shm-default-byte-size=10485760
```

>> **注意**: 当出现"Address already in use", 请使用`--grpc-port`指定端口号来启动服务，同时更改grpc_client.py中的请求端口号

服务启动成功后， 会有以下输出:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## 客户端请求
客户端请求可以在本地执行脚本请求；也可以在容器中执行。

本地执行脚本需要先安装依赖:
```
pip install grpcio
pip install tritonclient[all]

# 如果bash无法识别括号，可以使用如下指令安装:
pip install tritonclient\[all\]

# 发送请求
python3 grpc_client.py
```

发送请求成功后，会返回结果并打印输出:
```
1. Named Entity Recognition Task--------------
The extraction schema: ['时间', '选手', '赛事名称']
text= ['2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！']
results:
{'时间': {'end': 6,
        'probability': 0.9857379794120789,
        'start': 0,
        'text': '2月8日上午'},
 '赛事名称': {'end': 23,
          'probability': 0.8503087162971497,
          'start': 6,
          'text': '北京冬奥会自由式滑雪女子大跳台决赛'},
 '选手': {'end': 31,
        'probability': 0.8981545567512512,
        'start': 28,
        'text': '谷爱凌'}}
================================================
text= ['2月7日北京冬奥会短道速滑男子1000米决赛中任子威获得冠军！']
results:
{'时间': {'end': 4,
        'probability': 0.9921242594718933,
        'start': 0,
        'text': '2月7日'},
 '赛事名称': {'end': 22,
          'probability': 0.8171929121017456,
          'start': 4,
          'text': '北京冬奥会短道速滑男子1000米决赛'},
 '选手': {'end': 26,
        'probability': 0.9821093678474426,
        'start': 23,
        'text': '任子威'}}

2. Relation Extraction Task
The extraction schema: {'竞赛名称': ['主办方', '承办方', '已举办次数']}
text= ['2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。']
results:
{'竞赛名称': {'end': 13,
          'probability': 0.7825395464897156,
          'relation': {'主办方': [{'end': 22,
                                'probability': 0.8421710729598999,
                                'start': 14,
                                'text': '中国中文信息学会'},
                               {'end': 30,
                                'probability': 0.7580801248550415,
                                'start': 23,
                                'text': '中国计算机学会'}],
                       '已举办次数': [{'end': 82,
                                  'probability': 0.4671308398246765,
                                  'start': 80,
                                  'text': '4届'}],
                       '承办方': [{'end': 39,
                                'probability': 0.8292703628540039,
                                'start': 35,
                                'text': '百度公司'},
                               {'end': 55,
                                'probability': 0.7000497579574585,
                                'start': 40,
                                'text': '中国中文信息学会评测工作委员会'},
                               {'end': 72,
                                'probability': 0.6193480491638184,
                                'start': 56,
                                'text': '中国计算机学会自然语言处理专委会'}]},
          'start': 0,
          'text': '2022语言与智能技术竞赛'}}
```


## 配置修改

当前默认配置在GPU上运行Paddle引擎,如果要在CPU/GPU或其他推理引擎上运行, 需要修改配置，详情请参考[配置文档](../../../../serving/docs/zh_CN/model_configuration.md)

## 使用VisualDL进行可视化部署

可以使用VisualDL进行[Serving可视化部署](../../../../serving/docs/zh_CN/vdl_management.md)，上述启动服务、配置修改以及客户端请求的操作都可以基于VisualDL进行。

通过VisualDL的可视化界面对UIE进行服务化部署只需要如下三步：
```text
1. 载入模型库：./text/uie/serving/models
2. 下载模型资源文件：点击uie模型，点击版本号1添加预训练模型，选择信息抽取模型uie-base进行下载
3. 启动服务：点击启动服务按钮，输入启动参数。
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211709329-3261758e-69af-4efd-9711-693f5f031131.gif" width="100%"/>
</p>
