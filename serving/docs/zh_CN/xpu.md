# FastDeploy XPU Triton Server使用文档
FastDeploy XPU Triton Server通过Paddle Inference调用XPU进行推理，并且已经接入到 Triton Server。在FastDeploy XPU Triton Server中，使用XPU推理需要通过CPU instance_group和cpu_execution_accelerator进行配置和调用。本文档以PaddleClas为例，讲述如何把一个CPU/GPU Triton服务，改造成XPU Triton服务。

## 1. 准备服务化镜像  

- 下载FastDeploy XPU Triton Server镜像  
```bash
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.7-xpu-21.10  # 稳定版
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:0.0.0-xpu-21.10  # develop版本
```  

- 下载部署示例代码
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/serving

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

# 将配置文件放入预处理目录
mv ResNet50_vd_infer/inference_cls.yaml models/preprocess/1/inference_cls.yaml

# 将模型放入 models/runtime/1目录下, 并重命名为model.pdmodel和model.pdiparams
mv ResNet50_vd_infer/inference.pdmodel models/runtime/1/model.pdmodel
mv ResNet50_vd_infer/inference.pdiparams models/runtime/1/model.pdiparams
```

## 2. 启动容器
```bash
docker run -itd --name fd_xpu_server -v `pwd`/:/serving --net=host --privileged registry.baidubce.com/paddlepaddle/fastdeploy:1.0.7-xpu-21.10 /bin/bash
```

## 3. 验证XPU可用性
```bash
docker exec -it fd_xpu_server /bin/bash
cd /opt/fastdeploy/benchmark/cpp/build

# 设置XPU L3 Cache (R200是63Mb)
export XPU_PADDLE_L3_SIZE=67104768  
# 运行benchmark验证
./benchmark --model ResNet50_infer --config_path ../config/config.xpu.paddle.fp32.txt --enable_log_info

cd /serving
```
输出为：  
```
I0529 11:07:46.860354   222 memory_optimize_pass.cc:222] Cluster name : batch_norm_46.tmp_2_max  size: 1
--- Running analysis [ir_graph_to_program_pass]
I0529 11:07:46.889616   222 analysis_predictor.cc:1705] ======= optimize end =======
I0529 11:07:46.890262   222 naive_executor.cc:160] ---  skip [feed], feed -> inputs
I0529 11:07:46.890703   222 naive_executor.cc:160] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
[INFO] fastdeploy/runtime/runtime.cc(286)::CreatePaddleBackend	Runtime initialized with Backend::PDINFER in Device::KUNLUNXIN.
[INFO] fastdeploy/runtime/backends/paddle/paddle_backend.cc(341)::Infer	Running profiling for Runtime without H2D and D2H, Repeats: 1000, Warmup: 200
Runtime(ms): 0.706382ms.
```
显示启动的设备类型为：Device::KUNLUNXIN。FastDeploy Benchmark工具使用文档，请参考[benchmark](https://github.com/PaddlePaddle/FastDeploy/tree/develop/benchmark/cpp).

## 4. 配置Triton Model Config  
```protobuf
# XPU服务化案例: examples/vision/classification/serving/models/runtime/config.pbtxt
# 将XPU部分的注释撤销，并注释掉原来的GPU设置，修改为：
# # Number of instances of the model
# instance_group [
#   {
#     # The number of instances is 1
#     count: 1
#     # Use GPU, CPU inference option is:KIND_CPU
#     kind: KIND_GPU
#     # kind: KIND_CPU
#     # The instance is deployed on the 0th GPU card
#     gpus: [0]
#   }
# ]

# optimization {
#   execution_accelerators {
#   gpu_execution_accelerator : [ {
#     # use TRT engine
#     name: "tensorrt",
#     # use fp16 on TRT engine
#     parameters { key: "precision" value: "trt_fp16" }
#   },
#   {
#     name: "min_shape"
#     parameters { key: "inputs" value: "1 3 224 224" }
#   },
#   {
#     name: "opt_shape"
#     parameters { key: "inputs" value: "1 3 224 224" }
#   },
#   {
#     name: "max_shape"
#     parameters { key: "inputs" value: "16 3 224 224" }
#   }
#   ]
# }}

instance_group [
  {
    # The number of instances is 1
    count: 1
    # Use GPU, CPU inference option is:KIND_CPU
    # kind: KIND_GPU
    kind: KIND_CPU
    # The instance is deployed on the 0th GPU card
    # gpus: [0]
  }
]

optimization {
  execution_accelerators {
  cpu_execution_accelerator: [{
    name: "paddle_xpu",
    parameters { key: "cpu_threads" value: "4" }
    parameters { key: "use_paddle_log" value: "1" }
    parameters { key: "kunlunxin_id" value: "0" }
    parameters { key: "l3_workspace_size" value: "62914560" }
    parameters { key: "locked" value: "0" }
    parameters { key: "autotune" value: "1" }
    parameters { key: "precision" value: "int16" }
    parameters { key: "adaptive_seqlen" value: "0" }
    parameters { key: "enable_multi_stream" value: "0" }
    parameters { key: "gm_default_size" value: "0" }
    }]
}}
```

## 5. 启动Triton服务  
```bash
fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```  
输出：
```
[INFO] fastdeploy/runtime/runtime.cc(286)::CreatePaddleBackend	Runtime initialized with Backend::PDINFER in Device::KUNLUNXIN.
.....
I0529 03:54:40.585326 385 server.cc:592]
+-------------+---------+--------+
| Model       | Version | Status |
+-------------+---------+--------+
| paddlecls   | 1       | READY  |
| postprocess | 1       | READY  |
| preprocess  | 1       | READY  |
| runtime     | 1       | READY  |
+-------------+---------+--------+
......
I0529 03:54:40.586430 385 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0529 03:54:40.586657 385 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0529 03:54:40.627382 385 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

## 6. 客户端请求  
在物理机器中执行以下命令，发送grpc请求并输出结果:
```bash
# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 安装客户端依赖
python3 -m pip install tritonclient\[all\]

# 发送请求
python3 paddlecls_grpc_client.py
```

发送请求成功后，会返回json格式的检测结果并打印输出:
```bash
output_name: CLAS_RESULT
{'label_ids': [153], 'scores': [0.6858349442481995]}
```
以上测试结果为Paddle Inference Backend + XPU R200下的输出。

## 7. 容器内自测  
如果是想在容器内自测，则运行以下命令：  
```bash
cd /serving
# 后台挂载
nohup fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760 > log.txt 2>&1 &
# 安装客户端依赖
python3 -m pip install tritonclient\[all\]
# 发送请求
unset http_proxy
unset https_proxy
python3 paddlecls_grpc_client.py
```

## 8. 配置修改

当前默认配置在XPU运行Paddle Inference引擎， 如果要在CPU/GPU其他推理引擎上运行。 需要修改`models/runtime/config.pbtxt`中配置，详情请参考[配置文档](./model_configuration.md).

## 9. 常见问题
- [如何编写客户端 HTTP/GRPC 请求](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/client.md)
- [如何编译服务化部署镜像](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/compile.md)
- [服务化部署原理及动态Batch介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/demo.md)
- [模型仓库介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/serving/docs/zh_CN/model_repository.md)
