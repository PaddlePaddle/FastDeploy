English | [简体中文](README.md)
# PP-OCR Serving Deployment Example

Before the serving deployment, please confirm 

- 1.  Refer to [FastDeploy Serving Deployment](../../../../../serving/README_CN.md) for software and hardware environment requirements and image pull commands

## Introduction
This document describes how to build an OCR text recognition service with FastDeploy.

The server must be started in docker, while the client does not need to be in a docker container.

**The models in the path ($PWD) contain the model configuration and code (the server will load the models and code to start the service), which needs to be mapped to docker.**

OCR consists of det (detection), cls (classification) and rec (recognition) models.

The diagram of the serving deployment is shown below, where `pp_ocr` connects to `det_preprocess`、`det_runtime` and `det_postprocess`,`cls_pp` connects to `cls_runtime` and `cls_postprocess`,`rec_pp` connects to `rec_runtime` and `rec_postprocess`.

In particular, `cls_pp` and `rec_pp` services are called multiple times in `det_postprocess` to realize the classification and identification of the detection results (multiple boxes), and finally return the identification results to users.
<p align="center">
    <br>
<img src='./ppocr.png'">
    <br>
<p>

## Usage
### 1. Server
#### 1.1 Docker
```bash
# Download the repository code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/ocr/PP-OCRv3/serving/

# Dpwnload model, image, and dictionary files
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar xvf ch_PP-OCRv3_det_infer.tar && mv ch_PP-OCRv3_det_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/det_runtime/ && rm -rf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar && mv ch_ppocr_mobile_v2.0_cls_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/cls_runtime/ && rm -rf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xvf ch_PP-OCRv3_rec_infer.tar && mv ch_PP-OCRv3_rec_infer 1
mv 1/inference.pdiparams 1/model.pdiparams && mv 1/inference.pdmodel 1/model.pdmodel
mv 1 models/rec_runtime/ && rm -rf ch_PP-OCRv3_rec_infer.tar

mkdir models/pp_ocr/1 && mkdir models/rec_pp/1 && mkdir models/cls_pp/1

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt
mv ppocr_keys_v1.txt models/rec_postprocess/1/

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

# x.y.z represent the image version. Refer to serving document to replace them with numbers
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10
docker run -dit --net=host --name fastdeploy --shm-size="1g" -v $PWD:/ocr_serving registry.baidubce.com/paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 bash
docker exec -it -u root fastdeploy bash
```

#### 1.2 Installation (in docker)
```bash
ldconfig
apt-get install libgl1
```

#### 1.3 Start the server (in docker)
```bash
fastdeployserver --model-repository=/ocr_serving/models
```

Parameter:
  - `model-repository`(required): The storage path of the entire model streaming_pp_tts.
  - `http-port`(optional): Port number for the HTTP service. Default: `8000`. This port is not used in this example. 
  - `grpc-port`(optional): Port number for the GRPC service. Default: `8001`.
  - `metrics-port`(optional): Port number for the serer metric. Default: `8002`. This port is not used in this example.


### 2. Client
#### 2.1 Installation
```bash
pip3 install tritonclient[all]
```

#### 2.2 Send Requests
```bash
python3 client.py
```

## Configuration Change

The current default configuration runs on GPU. If you want to run it on CPU or other inference engines, please modify the configuration in `models/runtime/config.pbtxt`. Refer to [Configuration Document](../../../../../serving/docs/zh_CN/model_configuration.md) for more information.
