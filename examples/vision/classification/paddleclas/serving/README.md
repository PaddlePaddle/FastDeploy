English | [简体中文](README_CN.md)
# PaddleClas Service Deployment Example

Before the service deployment, please confirm 

- 1. Refer to [FastDeploy Service Deployment](../../../../../serving/README.md) for software and hardware environment requirements and image pull commands.


## Start the Service

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/classification/paddleclas/serving

# Download ResNet50_vd model files and test images 
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Put the configuration file into the preprocessing directory 
mv ResNet50_vd_infer/inference_cls.yaml models/preprocess/1/inference_cls.yaml

# Place the model under models/runtime/1 and rename them to model.pdmodel和model.pdiparams
mv ResNet50_vd_infer/inference.pdmodel models/runtime/1/model.pdmodel
mv ResNet50_vd_infer/inference.pdiparams models/runtime/1/model.pdiparams

# Pull the fastdeploy image (x.y.z represent the image version. Refer to the serving document to replace them with numbers)
# GPU image 
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU image 
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10

# Run the container named fd_serving and mount it in the /serving directory of the container 
nvidia-docker run -it --net=host --name fd_serving -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# Start the service (The CUDA_VISIBLE_DEVICES  environment variable is not set, which entitles the scheduling authority of all GPU cards)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models --backend-config=python,shm-default-byte-size=10485760
```
>> **Attention**:

>> To pull images from other hardware, refer to [Service Deployment Master Document](../../../../../serving/README.md)

>> If "Address already in use" appears when running fastdeployserver to start the service, use `--grpc-port` to specify the port number and change the request port number in the client demo.

>> Other startup parameters can be checked by fastdeployserver --help 

Successful service start brings the following output:
```
......
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## Client Request 

Execute the following command in the physical machine to send the grpc request and output the result
```
# Download test images 
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Install client dependencies 
python3 -m pip install tritonclient\[all\]

# Send the request 
python3 paddlecls_grpc_client.py
```

The result is returned in json format and printed after sending the request:
```
output_name: CLAS_RESULT
{'label_ids': [153], 'scores': [0.6862289905548096]}
```

## Configuration Change

The current default configuration runs the TensorRT engine on GPU. If you want to run it on CPU or other inference engines, please modify the configuration in `models/runtime/config.pbtxt`. Refer to [Configuration Document](../../../../../serving/docs/EN/model_configuration-en.md) for more information.
