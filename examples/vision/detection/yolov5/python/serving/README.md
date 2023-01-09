English | [简体中文](README_CN.md)

# YOLOv5 Python Simple Serving Demo


## Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Server:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/yolov5/python/serving

# Download model
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_infer.tar
tar xvf yolov5s_infer.tar

# Launch server, change the configurations in server.py to select hardware, backend, etc.
# and use --host, --port to specify IP and port
fastdeploy simple_serving --app server:app
```

Client:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/yolov5/python/serving

# Download test image
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Send request and get inference result (Please adapt the IP and port if necessary)
python client.py
```
