# FastDeploy 工具包
FastDeploy提供了一系列高效易用的工具优化部署体验, 提升推理性能.
例如, FastDeploy基于PaddleSlim的Auto Compression Toolkit(ACT), 给用户提供了一键模型自动化压缩的工具, 用户可以轻松地通过一行命令对模型进行自动化压缩. 本文档将以FastDeploy一键模型自动化压缩工具为例, 介绍如何安装此工具, 并提供相应的使用文档.

## FastDeploy一键模型自动化压缩工具

### 环境准备
1.用户参考PaddlePaddle官网, 安装develop版本
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.安装PaddleSlim develop版本
```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

3.安装fd-auto-compress一键模型自动化压缩工具
```bash
# 通过pip安装fd-auto-compress.
# FastDeploy的python包已包含此工具, 不需重复安装.
pip install fd-auto-compress

# 在当前目录执行以下命令
python setup.py install
```

### 一键模型自动化压缩工具的使用
按照以上步骤成功安装后,即可使用FastDeploy一键模型自动化压缩工具, 示例如下.

```bash
fastdeploy --auto_compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```
详细使用文档请参考[FastDeploy一键模型自动化压缩工具](./auto_compression/README.md)
