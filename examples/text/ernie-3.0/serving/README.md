# Ernie 3.0 服务化部署示例

## 准备模型

下载ERNIE 3.0的新闻分类模型、序列标注模型(如果有已训练好的模型，跳过此步骤):
```bash
# 下载并解压新闻分类模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_pruned_infer_model.zip
unzip tnews_pruned_infer_model.zip

# 将下载的模型移动到分类任务的模型仓库目录
mv tnews_pruned_infer_model/float32.pdmodel models/ernie_seqcls_model/1/model.pdmodel
mv tnews_pruned_infer_model/float32.pdiparams models/ernie_seqcls_model/1/model.pdiparams

# 下载并解压序列标注模型
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_pruned_infer_model.zip
unzip msra_ner_pruned_infer_model.zip

# 将下载的模型移动到序列标注任务的模型仓库目录
mv msra_ner_pruned_infer_model/float32.pdmodel models/ernie_tokencls_model/1/model.pdmodel
mv msra_ner_pruned_infer_model/float32.pdiparams models/ernie_tokencls_model/1/model.pdiparams
```

模型下载移动好之后，分类任务的models目录结构如下:
```
models
├── ernie_seqcls                      # 分类任务的pipeline
│   ├── 1
│   └── config.pbtxt                  # 通过这个文件组合前后处理和模型推理
├── ernie_seqcls_model                # 分类任务的模型推理
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── ernie_seqcls_postprocess          # 分类任务后处理
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                   # 预处理分词
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## 拉取并运行镜像
```bash
# CPU镜像, 仅支持Paddle/ONNX模型在CPU上进行服务化部署，支持的推理后端包括OpenVINO、Paddle Inference和ONNX Runtime
docker pull paddlepaddle/fastdeploy:0.3.0-cpu-only-21.10

# GPU 镜像, 支持Paddle/ONNX模型在GPU/CPU上进行服务化部署，支持的推理后端包括OpenVINO、TensorRT、Paddle Inference和ONNX Runtime
docker pull paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10

# 运行
docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models paddlepaddle/fastdeploy:0.3.0-cpu-only-21.10 bash
```

## 部署模型
serving目录包含启动pipeline服务的配置和发送预测请求的代码，包括：

```
models                    # 服务化启动需要的模型仓库，包含模型和服务配置文件
seq_cls_rpc_client.py     # 新闻分类任务发送pipeline预测请求的脚本
token_cls_rpc_client.py   # 序列标注任务发送pipeline预测请求的脚本
```

*注意*:启动服务时，Server的每个python后端进程默认申请`64M`内存，默认启动的docker无法启动多个python后端节点。有两个解决方案：
- 1.启动容器时设置`shm-size`参数, 比如:`docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10 bash`
- 2.启动服务时设置python后端的`shm-default-byte-size`参数, 设置python后端的默认内存为10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

### 分类任务
在容器内执行下面命令启动服务:
```
# 默认启动models下所有模型
fastdeployserver --model-repository=/models

# 可通过参数只启动分类任务
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_seqcls
```
输出打印如下:
```
I1019 09:41:15.375496 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_tokenizer' version 1
I1019 09:41:15.375987 2823 model_repository_manager.cc:1022] loading: ernie_seqcls:1
I1019 09:41:15.477147 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_seqcls' version 1
I1019 09:41:15.477325 2823 server.cc:522]
...
I0613 08:59:20.577820 10021 server.cc:592]
+----------------------------+---------+--------+
| Model                      | Version | Status |
+----------------------------+---------+--------+
| ernie_seqcls               | 1       | READY  |
| ernie_seqcls_model         | 1       | READY  |
| ernie_seqcls_postprocess   | 1       | READY  |
| ernie_tokenizer            | 1       | READY  |
+----------------------------+---------+--------+
...
I0601 07:15:15.923270 8059 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0601 07:15:15.923604 8059 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0601 07:15:15.964984 8059 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### 序列标注任务
在容器内执行下面命令启动序列标注服务:
```
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_tokencls --backend-config=python,shm-default-byte-size=10485760
```
输出打印如下:
```
I1019 09:41:15.375496 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_tokenizer' version 1
I1019 09:41:15.375987 2823 model_repository_manager.cc:1022] loading: ernie_seqcls:1
I1019 09:41:15.477147 2823 model_repository_manager.cc:1183] successfully loaded 'ernie_seqcls' version 1
I1019 09:41:15.477325 2823 server.cc:522]
...
I0613 08:59:20.577820 10021 server.cc:592]
+----------------------------+---------+--------+
| Model                      | Version | Status |
+----------------------------+---------+--------+
| ernie_tokencls             | 1       | READY  |
| ernie_tokencls_model       | 1       | READY  |
| ernie_tokencls_postprocess | 1       | READY  |
| ernie_tokenizer            | 1       | READY  |
+----------------------------+---------+--------+
...
I0601 07:15:15.923270 8059 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0601 07:15:15.923604 8059 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0601 07:15:15.964984 8059 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## 客户端请求
客户端请求可以在本地执行脚本请求；也可以在容器中执行。

本地执行脚本需要先安装依赖:
```
pip install grpcio
pip install tritonclient[all]

# 如果bash无法识别括号，可以使用如下指令安装:
pip install tritonclient\[all\]
```

### 分类任务
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)
```
python seq_cls_grpc_client.py
```
输出打印如下:
```
{'label': array([5, 9]), 'confidence': array([0.6425664 , 0.66534853], dtype=float32)}
{'label': array([4]), 'confidence': array([0.53198355], dtype=float32)}
acc: 0.5731
```

### 序列标注任务
注意执行客户端请求时关闭代理，并根据实际情况修改main函数中的ip地址(启动服务所在的机器)
```
python token_cls_grpc_client.py
```
输出打印如下:
```
input data: 北京的涮肉，重庆的火锅，成都的小吃都是极具特色的美食。
The model detects all entities:
entity: 北京   label: LOC   pos: [0, 1]
entity: 重庆   label: LOC   pos: [6, 7]
entity: 成都   label: LOC   pos: [12, 13]
input data: 原产玛雅故国的玉米，早已成为华夏大地主要粮食作物之一。
The model detects all entities:
entity: 玛雅   label: LOC   pos: [2, 3]
entity: 华夏   label: LOC   pos: [14, 15]
```

## 配置修改

当前分类任务(ernie_seqcls_model/config.pbtxt)默认配置在CPU上运行OpenVINO引擎; 序列标注任务默认配置在GPU上运行Paddle引擎。如果要在CPU/GPU或其他推理引擎上运行, 需要修改配置，详情请参考[配置文档](../../../../../serving/docs/zh_CN/model_configuration.md)
