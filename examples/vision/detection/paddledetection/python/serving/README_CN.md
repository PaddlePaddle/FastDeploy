简体中文 | [English](README_EN.md)

# PaddleDetection Python轻量服务化部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

服务端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/python/serving

# 下载PPYOLOE模型文件（如果不下载，代码里会自动从hub下载）
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz

# 安装uvicorn
pip install uvicorn

# 启动服务，可选择是否使用GPU和TensorRT，可根据uvicorn --help配置IP、端口号等
# CPU
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=cpu uvicorn server:app
# GPU
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=gpu uvicorn server:app
# GPU上使用TensorRT （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
MODEL_DIR=ppyoloe_crn_l_300e_coco DEVICE=gpu USE_TRT=true uvicorn server:app
```

客户端：
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/python/serving

# 下载测试图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 请求服务，获取推理结果（如有必要，请修改脚本中的IP和端口号）
python client.py
```
