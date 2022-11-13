# FastDeploy 工具包
FastDeploy工具包提供例如一键模型自动化压缩等工具, 本文降介绍如何使用此工具包

## 以FastDeploy一键模型压缩为例

### 环境准备
1.用户参考PaddlePaddle官网, 安装develop版本
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.安装paddleslim-develop版本
```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

3.安装fd-auto-compress一键模型自动化压缩工具
```bash
# 通过pip安装fd-auto-compress
pip install fd-auto-compress

# 并在当前目录执行
python setup.py install
```
### 一键量化工具的使用
按照以上步骤成功安装后,即可使用FastDeploy一键模型自动化压缩工具, 示例如下.
fastdeploy --auto_compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
详细使用文档请参考[FastDeploy一键模型自动化压缩工具](./auto_compression/README.md)
