English | [简体中文](README_CN.md)
# PaddleDetection Serving Deployment Example

This document gives a detailed introduction to the deployment of PP-YOLOE models(ppyoloe_crn_l_300e_coco). Other PaddleDetection model all support serving deployment. So users just need to change the model and config name in the following command.

For PaddleDetection model export and download of pre-trained models, refer to [PaddleDetection Model Deployment](../README.md).

Confirm before the serving deployment

- 1. Refer to [FastDeploy Serving Deployment](../../../../../serving/README.md) for software and hardware environment requirements and image pull commands


## Start Service

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/serving

# Download PPYOLOE model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# Put the configuration file into the preprocessing directory
mv ppyoloe_crn_l_300e_coco/infer_cfg.yml models/preprocess/1/

# Place the model under models/runtime/1 and rename them to model.pdmodel and model.pdiparams
mv ppyoloe_crn_l_300e_coco/model.pdmodel models/runtime/1/model.pdmodel
mv ppyoloe_crn_l_300e_coco/model.pdiparams models/runtime/1/model.pdiparams

# Rename the ppyoloe config files in ppdet and runtime to standard config names
# For other models like faster_rcc, rename faster_rcnn_config.pbtxt to config.pbtxt
cp models/ppdet/ppyoloe_config.pbtxt models/ppdet/config.pbtxt
cp models/runtime/ppyoloe_runtime_config.pbtxt models/runtime/config.pbtxt

# Attention: Given that the mask_rcnn model has one more output, we need to rename mask_config.pbtxt to config.pbtxt in the postprocess directory (models/postprocess)

# Pull the FastDeploy image (x.y.z represent the image version. Users need to replace them with numbers)
# GPU image
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
# CPU image
docker pull paddlepaddle/fastdeploy:z.y.z-cpu-only-21.10


# Run the container named fd_serving and mount it in the /serving directory of the container
nvidia-docker run -it --net=host --name fd_serving --shm-size="1g"  -v `pwd`/:/serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10  bash

# Start Service (The CUDA_VISIBLE_DEVICES environment variable is not set, which entitles the scheduling authority of all GPU cards)
CUDA_VISIBLE_DEVICES=0 fastdeployserver --model-repository=/serving/models
```
>> **Attention**:

>> Given that the mask_rcnn model has one more output, we need to rename mask_config.pbtxt to config.pbtxt in the postprocess directory (models/postprocess)

>> To pull images, refer to [Service Deployment Master Document](../../../../../serving/README_CN.md)

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

Execute the following command in the physical machine to send the grpc request and output the results
```
# Download test images
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Install client dependencies
python3 -m pip install tritonclient[all]

# Send requests
python3 paddledet_grpc_client.py
```

The result is returned in json format and printed after sending the request:
```
output_name: DET_RESULT
[[159.93016052246094, 82.35527038574219, 199.8546600341797, 164.68682861328125],
... ...,
[60.200584411621094, 123.73260498046875, 108.83859252929688, 169.07467651367188]]
```

## Configuration Change

The current default configuration runs on GPU. If you want to run it on CPU or other inference engines, please modify the configuration in `models/runtime/config.pbtxt`. Refer to [Configuration Document](../../../../../serving/docs/EN/model_configuration-en.md) for more information.

## Use VisualDL for serving deployment visualization

You can use VisualDL for [serving deployment visualization](../../../../../serving/docs/EN/vdl_management-en.md) , the above model preparation, deployment, configuration modification and client request operations can all be performed based on VisualDL.

The serving deployment of PaddleDetection by VisualDL only needs the following three steps:
```text
1. Load the model repository: ./vision/detection/paddledetection/serving/models
2. Download the model resource file: click the preprocess model, click the version number 1 to add the pre-training model, and select the detection model ppyoloe_crn_l_300e_coco to download. click the runtime model, click the version number 1 to add the pre-training model, and select the detection model ppyoloe_crn_l_300e_coco to download.
3. Set startup config file: click the "ensemble  configuration" button, choose configuration file ppyoloe_config.pbtxt, then click the "set as startup config" button. click the runtime model, choose configuration file ppyoloe_runtime_config.pbtxt, then click the "set as startup config" button.
4. Start the service: Click the "launch server" button and input the launch parameters.
```
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/211710983-2d1f1427-6738-409d-903b-2b4e4ab6cbfc.gif" width="100%"/>
</p>
