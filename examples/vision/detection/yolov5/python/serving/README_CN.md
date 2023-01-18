简体中文 | [English](README.md)

# YOLOv5 Python轻量服务化部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

服务端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/yolov5/python/serving

# 下载模型文件
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_infer.tar
tar xvf yolov5s_infer.tar

# 启动服务，可修改server.py中的配置项来指定硬件、后端等
# 可通过--host、--port指定IP和端口号
fastdeploy simple_serving --app server:app
```

客户端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/yolov5/python/serving

# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 请求服务，获取推理结果（如有必要，请修改脚本中的IP和端口号）
python client.py
```
