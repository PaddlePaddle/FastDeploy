English | [简体中文](README_CN.md)

# Example of UIE Serving Deployment

Before serving deployment, you need to confirm:

- 1. You can refer to [FastDeploy Serving](../../../../../serving/README_CN.md) for hardware and software environment requirements and image pull commands for serving images.

## Prepare models

Download the UIE-Base model (if you have trained models, skip this step):
```bash
# Download UIE model documents and vocabulary. Taking the uie-base model as an example
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz

# Move the model to the model repository directory
mv uie-base/* models/uie/1/
```

After download and move, the models directory is as follows:

```
models
└── uie
    ├── 1
    │   ├── inference.pdiparams
    │   ├── inference.pdmodel
    │   ├── model.py
    │   └── vocab.txt
    └── config.pbtxt
```

## Pull and run images.
```bash
# x.y.z represent image versions. You can refer to the serving documents to replace them with numbers
# GPU Image
docker pull paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU Image
docker pull paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# Run the container. The container name is fd_serving, and the current directory is mounted as the container's /uie_serving directory
docker run  -it --net=host --name fastdeploy_server --shm-size="1g" -v `pwd`/:/uie_serving paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash

# reinstall fastdeploy python
# gpu
python3 -m pip uninstall fastdeploy-gpu-python
python3 -m pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

# cpu
python3 -m pip uninstall fastdeploy-python
python3 -m pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

# Start the service (it will have scheduling privileges for all GPU cards without setting the CUDA_VISIBLE_DEVICES environment variable)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/uie_serving/models --backend-config=python,shm-default-byte-size=10485760
```

>> **Attention**: When appearing "Address already in use", please use `--grpc-port`specified port numbers to start the service. Meanwhile you should change the requesting port numbers in grpc_client.py

When starting the service, the following output will be displayed:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## Client Requests
Client requests can execute script requests locally and in the container.

Dependencies should be installed to execute the script locally:
```
pip install grpcio
pip install tritonclient[all]

# If bash cannot recognize the brackets, you can use the following command to install dependencies:
pip install tritonclient\[all\]

# Send Requests
python3 grpc_client.py
```

When the request is sent successfully, the result is returned and printed out:
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


## Configuration Modification

The current configuration is by default to run the paddle engine on CPU. If you want to run on CPU/GPU or other inference engines, modifying the configuration is needed.Please refer to [Config Doc](../../../../serving/docs/zh_CN/model_configuration.md)
