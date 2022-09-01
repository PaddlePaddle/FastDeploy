# FastDeploy 预编编译Python Wheel包

FastDeploy提供了在Windows/Linux/Mac上的预先编译Python Wheel包，开发者可以直接下载后安装，也可以自行编译代码。

目前各平台支持情况如下

- Linux 支持Python3.6~3.9
- Windows 支持Python3.6~3.9
- Mac 支持Python3.6~3.9

## 安装 CPU Python 版本
```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
## 安装 GPU Python 版本
```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## Anaconda 快速配置 GPU 环境
使用Anaconda的用户可在命令运行以下命令，快速配置GPU环境。如果是Windows用户，需要先打开`Anaconda Prompt (anaconda3)`命令行终端。
- 增加 conda-forge 源
```bash
conda config --add channels conda-forge
# 国内用户可以增加国内的源，如
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
```
- 新建 python 环境
```bash
conda create -n py38 python=3.8
conda activate py38
```  
- 安装 cudatoolkit 11.x 和 cudnn 8.x
```bash
conda install cudatoolkit=11.2 cudnn=8.2
```
- 安装 FastDeploy GPU 版本 Python 包
```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## 注意事项
- 不要重复安装`fastdeploy-python`和`fastdeploy-gpu-python`
- 如已安装CPU版本的`fastdeploy-python`后，在安装GPU版本的`fastdeploy-gpu-python`，请先执行`pip uninstall fastdeploy-python`卸载已有版本

## 环境依赖

- cuda >= 11.2
- cudnn >= 8.0

## 其它文档

- [预编译C++部署库](./CPP_prebuilt_libraries.md)
- [视觉模型C++/Python部署示例](../../examples/vision/)
