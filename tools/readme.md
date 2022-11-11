# FastDeploy 工具包
FastDeploy工具包提供例如一键模型自动化压缩等工具, 本文降介绍如何使用此工具包

## 以FastDeploy一键模型压缩为例

### 环境准备
1.装paddles
2.paddleslim
3.fastdeploy-auto-compression

### 安装工具包
python setup.py install

### 使用FastDeploy一键模型压缩工具
fastdeploy --auto_compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
