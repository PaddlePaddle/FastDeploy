English | [简体中文](README_CN.md)

# PaddleDetection Python Simple Serving Demo


## Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Server:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/python/serving

# Download PPYOLOE model
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz

# Install uvicorn
pip install uvicorn

# Launch server, it's configurable to use GPU and TensorRT,
# and run 'uvicorn --help' to check how to specify IP and port, etc.
# CPU
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=cpu uvicorn server:app
# GPU
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=gpu uvicorn server:app
# GPU and TensorRT
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=gpu USE_TRT=true uvicorn server:app
```

Client:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/python/serving

# Download test image
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Send request and get inference result (Please adapt the IP and port if necessary)
python client.py
```
