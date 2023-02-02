English | [简体中文](README_CN.md)

# Example of ERNIE 3.0 Serving Deployment

Before serving deployment, you need to confirm

- 1. Refer to [FastDeploy Serving Deployment](../../../../serving/README.md) for hardware and software environment requirements and image pull commands of serving images.

## Prepare Models

Download the news classification model and the sequence labeling model of ERNIE 3.0 (if you have trained models, skip this step):
```bash
# Download and decompress the news classification model
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/tnews_pruned_infer_model.zip
unzip tnews_pruned_infer_model.zip

# Move the download model to the model repository directory of classification tasks.
mv tnews_pruned_infer_model/float32.pdmodel models/ernie_seqcls_model/1/model.pdmodel
mv tnews_pruned_infer_model/float32.pdiparams models/ernie_seqcls_model/1/model.pdiparams

# Download and decompress the sequence labelling model
wget https://paddlenlp.bj.bcebos.com/models/transformers/ernie_3.0/msra_ner_pruned_infer_model.zip
unzip msra_ner_pruned_infer_model.zip

# Move the download model to the model repository directory of sequence labeling task.
mv msra_ner_pruned_infer_model/float32.pdmodel models/ernie_tokencls_model/1/model.pdmodel
mv msra_ner_pruned_infer_model/float32.pdiparams models/ernie_tokencls_model/1/model.pdiparams
```

After download and move, the models directory of the classification tasks is as follows:
```
models
├── ernie_seqcls                      # Pipeline for classification task
│   ├── 1
│   └── config.pbtxt                  # Combine pre and post processing and model inference
├── ernie_seqcls_model                # Model inference for classification task
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
├── ernie_seqcls_postprocess          # Post-processing of classification task
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── ernie_tokenizer                   # Pre-processing splitting
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## Pull and Run Images
```bash
# x.y.z represent image versions. Please refer to the serving document to replace them with numbers
# GPU Image
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU Image
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# Running
docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10 bash
```

## Deployment Models
The serving directory contains the configuration to start the pipeline service and the code to send the prediction request, including

```
models                    # Model repository needed for serving startup, containing model and service configuration files
seq_cls_rpc_client.py     # Script for sending pipeline prediction requests for news classification task
token_cls_rpc_client.py   # Script for sequence annotation task to send pipeline prediction requests
```

*Attention*:Attention: When starting the service, each python backend process of Server requests 64M memory by default, and the docker started by default cannot start more than one python backend node. There are two solutions:

- 1.Set the `shm-size` parameter when starting the container, for example, `docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v /path/serving/models:/models registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash`
- 2.Set the `shm-default-byte-size` parameter of python backend when starting the service. Set the default memory of python backend to 10M： `tritonserver --model-repository=/models --backend-config=python,shm-default-byte-size=10485760`

### Classification Task
Execute the following command in the container to start the service:
```
# Enable all models by default
fastdeployserver --model-repository=/models

# You can only enable classification task via parameters
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_seqcls
```
The output is:
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

### Sequence Labelling Task
Execute the following command in the container to start the sequence labelling service:
```
fastdeployserver --model-repository=/models --model-control-mode=explicit --load-model=ernie_tokencls --backend-config=python,shm-default-byte-size=10485760
```
The output is:
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

## Client Requests
Client requests can execute script requests locally and in the container.

Dependencies should be installed to execute the script locally:
```
pip install grpcio
pip install tritonclient[all]

# If bash cannot recognize the brackets, you can use the following command to install dependencies:
pip install tritonclient\[all\]
```

### Classification Task
Attention: The proxy need turning off when executing client requests. The ip address in the main function (the machine where you start services) should be modified as appropriate.
```
python seq_cls_grpc_client.py
```
The output is:
```
{'label': array([5, 9]), 'confidence': array([0.6425664 , 0.66534853], dtype=float32)}
{'label': array([4]), 'confidence': array([0.53198355], dtype=float32)}
acc: 0.5731
```

### Sequence Labeling Task
Attention: The proxy need turning off when executing client requests. The ip address in the main function (the machine where you start services) should be modified as appropriate.
```
python token_cls_grpc_client.py
```
The output is:
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

## Configuration Modification
The current classification task (ernie_seqcls_model/config.pbtxt) is by default configured to run the OpenVINO engine on CPU; the sequence labelling task is by default configured to run the Paddle engine on GPU. If you want to run on CPU/GPU or other inference engines, you should modify the configuration. please refer to the [configuration document.](../../../../serving/docs/EN/model_configuration-en.md)

## Use VisualDL for serving deployment visualization

You can use VisualDL for [serving deployment visualization](../../../../serving/docs/EN/vdl_management-en.md) , the above model preparation, deployment, configuration modification and client request operations can all be performed based on VisualDL.

The serving deployment of ERNIE 3.0 by VisualDL only needs the following three steps:
```text
1. Load the model repository: ./text/ernie-3.0/serving/models
2. Download the model resource file: click the ernie_seqcls_model model, click the version number 1 to add the pre-training model, and select the text classification model ernie_3.0_ernie_seqcls_model to download. Click the ernie_tokencls_model model, click the version number 1 to add the pre-training model, and select the text classification model ernie_tokencls_model to download.
3. Start the service: Click the "launch server" button and input the launch parameters.
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211708353-507d6038-b754-4520-884b-1156703a44c6.gif" width="100%"/>
</p>
