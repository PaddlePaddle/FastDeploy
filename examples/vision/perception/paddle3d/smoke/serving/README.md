English | [简体中文](README_CN.md)
# Smoke service deployment example

This document introduces the service deployment of the Smoke model in Paddle3D.

For Smoke model export and pre-training model download, please refer to [Smoke Model Deployment](../README.md) document.

Before service deployment, you need to confirm

- 1. Please refer to [FastDeploy Service-based Deployment](../../../../../serving/README_CN.md) for the hardware and software environment requirements of the service image and the image pull command


## Start the service

```bash
#Download deployment sample code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/perception/paddle3d/smoke/serving

#Download the Smoke model file and test image
wget https://bj.bcebos.com/fastdeploy/models/smoke.tar.gz
tar -xf smoke.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# Put the configuration file into the preprocessing directory
mv smoke/infer_cfg.yml models/preprocess/1/

# Put the model into the models/runtime/1 directory, and rename it to model.pdmodel and model.pdiparams
mv smoke/smoke.pdmodel models/runtime/1/model.pdmodel
mv smoke/smoke.pdiparams models/runtime/1/model.pdiparams

# Pull the fastdeploy image (x.y.z is the image version number, which needs to be replaced with the fastdeploy version number)
# GPU mirroring
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU mirroring
docker pull paddlepaddle/fastdeploy:z.y.z-cpu-only-21.10


# Run the container. The container name is fd_serving, and mount the current directory as the /serving directory of the container
nvidia-docker run -it --net=host --name fd_serving --shm-size="1g" -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4 -trt8.4-21.10 bash

# Start the service (if you do not set the CUDA_VISIBLE_DEVICES environment variable, you will have the scheduling authority of all GPU cards)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models
```
>> **NOTE**:

>> For pulling images, please refer to [Serving Deployment Main Document](../../../../../serving/README_CN.md)

>> Execute fastdeployserver to start the service and "Address already in use" appears, please use `--grpc-port` to specify the grpc port number to start the service, and change the request port number in the client example.

>> Other startup parameters can be viewed using fastdeployserver --help

After the service starts successfully, there will be the following output:
```
 …
I0928 04:51:15.784517 206 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I0928 04:51:15.785177 206 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I0928 04:51:15.826578 206 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```


## Client request

Execute the following command on the physical machine, send the grpc request and output the result
```
#Download test image
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# Install client dependencies
python3 -m pip install tritonclient[all]

# send request
python3 smoke_grpc_client.py
```

After sending the request successfully, the detection result in json format will be returned and printed out:
```
output_name: PERCEPTION_RESULT
, 0.0068892366252839565]
label_ids: [2, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 , 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 , 0]
```

## Configuration modification

The current default configuration runs the Paddle engine on the CPU, if you want to run it on the GPU or other inference engines. It is necessary to modify the configuration in `models/runtime/config.pbtxt`, for details, please refer to [configuration document](../../../../../serving/docs/zh_CN/model_configuration.md)
