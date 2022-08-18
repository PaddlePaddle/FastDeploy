# FastDeploy 预编编译Python Wheel包

FastDeploy提供了在Windows/Linux/Mac上的预先编译Python Wheel包，开发者可以直接下载后安装，也可以自行编译代码。

目前各平台支持情况如下

- Linux 支持Python3.6~3.9
- Windows 支持Python3.8~3.9
- Mac 支持Python3.6~3.9

## 安装 CPU Python 版本
```
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
## 安装 GPU Python 版本
```
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
