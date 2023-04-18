[English](README.md) | 简体中文
# Smoke 服务化部署示例

本文档介绍 Paddle3D 中 Smoke 模型的服务化部署。

Smoke 模型导出和预训练模型下载请看[Smoke模型部署](../README.md)文档。

在服务化部署前，需确认

- 1. 服务化镜像的软硬件环境要求和镜像拉取命令请参考[FastDeploy服务化部署](../../../../../serving/README_CN.md)


## 启动服务

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/perception/paddle3d/smoke/serving

#下载 Smoke 模型文件和测试图片
wget https://bj.bcebos.com/fastdeploy/models/smoke.tar.gz
tar -xf smoke.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# 将配置文件放入预处理目录
mv smoke/infer_cfg.yml models/preprocess/1/

# 将模型放入 models/runtime/1 目录下, 并重命名为 model.pdmodel 和 model.pdiparams
mv smoke/smoke.pdmodel models/runtime/1/model.pdmodel
mv smoke/smoke.pdiparams models/runtime/1/model.pdiparams

# 拉取 fastdeploy 镜像(x.y.z 为镜像版本号，需替换成 fastdeploy 版本数字)
# GPU 镜像
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU 镜像
docker pull paddlepaddle/fastdeploy:z.y.z-cpu-only-21.10


# 运行容器.容器名字为 fd_serving, 并挂载当前目录为容器的 /serving 目录
nvidia-docker run -it --net=host --name fd_serving --shm-size="1g"  -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# 启动服务(不设置 CUDA_VISIBLE_DEVICES 环境变量，会拥有所有 GPU 卡的调度权限)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models
```
>> **注意**:

>> 拉取镜像请看[服务化部署主文档](../../../../../serving/README_CN.md)

>> 执行 fastdeployserver 启动服务出现 "Address already in use", 请使用 `--grpc-port` 指定 grpc 端口号来启动服务，同时更改客户端示例中的请求端口号.

>> 其他启动参数可以使用 fastdeployserver --help 查看

服务启动成功后， 会有以下输出:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## 客户端请求

在物理机器中执行以下命令，发送 grpc 请求并输出结果
```
#下载测试图片
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

#安装客户端依赖
python3 -m pip install tritonclient[all]

# 发送请求
python3 smoke_grpc_client.py
```

发送请求成功后，会返回 json 格式的检测结果并打印输出:
```
output_name: PERCEPTION_RESULT
scores:  [0.8080164790153503, 0.03356542810797691, 0.03165825456380844, 0.020817330107092857, 0.018075695261359215, 0.017861749976873398, 0.016441335901618004, 0.01476177480071783, 0.012927377596497536, 0.012407636269927025, 0.012400650419294834, 0.012216777540743351, 0.01208423636853695, 0.011721019633114338, 0.011697308160364628, 0.011695655062794685, 0.011603309772908688, 0.011140472255647182, 0.010927721858024597, 0.01036786288022995, 0.00984608568251133, 0.009827949106693268, 0.009761993773281574, 0.00959752406924963, 0.009595031850039959, 0.009423951618373394, 0.008946355432271957, 0.008635037578642368, 0.008597995154559612, 0.008552121929824352, 0.00839947909116745, 0.008325068280100822, 0.00830004084855318, 0.00826205126941204, 0.008174785412847996, 0.008085251785814762, 0.008026468567550182, 0.00796759407967329, 0.007873599417507648, 0.007816540077328682, 0.007742374204099178, 0.007734378334134817, 0.0077047450467944145, 0.007684454321861267, 0.007525254040956497, 0.007521109655499458, 0.007519087754189968, 0.007399206515401602, 0.0071790567599236965, 0.0068892366252839565]
label_ids:  [2, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## 配置修改

当前默认配置在 CPU 上运行 Paddle 引擎， 如果要在 GPU 或其他推理引擎上运行。 需要修改 `models/runtime/config.pbtxt` 中配置，详情请参考[配置文档](../../../../../serving/docs/zh_CN/model_configuration.md)
