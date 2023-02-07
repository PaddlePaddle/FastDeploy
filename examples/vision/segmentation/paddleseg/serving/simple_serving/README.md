English | [简体中文](README_CN.md)

# PaddleSegmentation Python Simple Serving Demo


## Environment

- 1. Prepare environment and install FastDeploy Python whl, refer to [download_prebuilt_libraries](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

Server:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python/serving

# Download PP_LiteSeg model
wget  https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_with_argmax_infer.tgz

# Launch server, change the configurations in server.py to select hardware, backend, etc.
# and use --host, --port to specify IP and port
fastdeploy simple_serving --app server:app
```

Client:
```bash
# Download demo code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python/serving

# Download test image
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# Send request and get inference result (Please adapt the IP and port if necessary)
python client.py
```
