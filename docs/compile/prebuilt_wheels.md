# FastDeploy 预编编译Python Wheel包

FastDeploy提供了在Windows/Linux/Mac上的预先编译Python Wheel包，开发者可以直接下载后安装，也可以自行编译代码。

使用如下命令即可在Linux的Pythotn 3.8环境安装支持CPU部署的FastDeploy
```
python -m pip install fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl
```

## 注意事项
- 不要重复安装`fastdeploy-python`和`fastdeploy-gpu-python`
- 如已安装CPU版本的`fastdeploy-python`后，在安装GPU版本的`fastdeploy-gpu-python`，请先执行`pip uninstall fastdeploy-python`卸载已有版本

## 环境依赖

- cuda >= 11.2
- cudnn >= 8.0

## 下载地址

### Linux x64平台

| CPU 安装包 | 硬件 | Python版本 |
| :------------- | :--- | :--------- |
| [fastdeploy_python-0.2.0-cp36-cp36m-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp36-cp36m-manylinux1_x86_64.whl) | CPU | 3.6 |
| [fastdeploy_python-0.2.0-cp37-cp37m-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp37-cp37m-manylinux1_x86_64.whl) | CPU | 3.7 |
| [fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl) | CPU | 3.8 |
| [fastdeploy_python-0.2.0-cp39-cp39-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_python-0.2.0-cp39-cp39-manylinux1_x86_64.whl) | CPU | 3.9 |

| GPU 安装包 | 硬件 | Python版本 |
| :------------- | :--- | :--------- |
| [fastdeploy_gpu_python-0.2.0-cp36-cp36m-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_gpu_python-0.2.0-cp36-cp36m-manylinux1_x86_64.whl) | CPU/GPU | 3.6 |
| [fastdeploy_gpu_python-0.2.0-cp37-cp37m-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_gpu_python-0.2.0-cp37-cp37m-manylinux1_x86_64.whl) | CPU/GPU | 3.7 |
| [fastdeploy_gpu_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_gpu_python-0.2.0-cp38-cp38-manylinux1_x86_64.whl) | CPU/GPU | 3.8 |
| [fastdeploy_gpu_python-0.2.0-cp39-cp39-manylinux1_x86_64.whl](https://bj.bcebos.com/paddlehub/fastdeploy/wheels/fastdeploy_gpu_python-0.2.0-cp39-cp39-manylinux1_x86_64.whl) | CPU/GPU | 3.9 |

### Windows 10 x64平台

| CPU 安装包 | 硬件 | Python版本 |
| :----  | :-- | :------ |
| [comming...] | CPU | 3.8 |
| [comming...] | CPU | 3.9 |

### Linux aarch64平台

| 安装包 | 硬件 | Python版本 |
| :----  | :-- | :------ |
| [comming...] | CPU | 3.7 |
| [comming...] | CPU | 3.8 | 
| [comming...] | CPU | 3.9 |

### Mac OSX平台

| 架构 | 硬件 | 安装包 | Python版本 |
| :----  | :-- | :------ | :----- |
| x86_64 | CPU | [comming...] | 3.9 |
| arm64 | CPU | [comming...] | 3.9 |

## 其它文档

- [预编译C++部署库](./prebuilt_libraries.md)
- [视觉模型C++/Python部署示例](../../examples/vision/)
