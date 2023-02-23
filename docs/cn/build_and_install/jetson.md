[English](../../en/build_and_install/jetson.md) | 简体中文

# Jetson部署库编译

FastDeploy当前在Jetson仅支持ONNX Runtime CPU和TensorRT GPU/Paddle Inference三种后端推理

- 如若编译过程，出现错误提示`Could not find a package configuration file provided by "Python" with any of the following names: PythonConfig.cmake python-config.cmake`，请尝试将[cmake升级至3.25或最新版本](https://cmake.org/download/)解决。

## C++ SDK编译安装

编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.10.0
- jetpack >= 4.6.1


如果需要集成Paddle Inference后端，在[Paddle Inference预编译库](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#c)页面根据开发环境选择对应的Jetpack C++包下载，并解压。

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DBUILD_ON_JETSON=ON \
         -DENABLE_VISION=ON \
         -DENABLE_PADDLE_BACKEND=ON \ # 可选项，如若不需要Paddle Inference后端，可关闭
         -DPADDLEINFERENCE_DIRECTORY=/Download/paddle_inference_jetson \
         -DCMAKE_INSTALL_PREFIX=${PWD}/installed_fastdeploy
make -j8
make install
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库


## Python编译安装

编译过程同样需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.10.0
- jetpack >= 4.6.1
- python >= 3.6

Python打包依赖`wheel`，编译前请先执行`pip install wheel`

如果需要集成Paddle Inference后端，在[Paddle Inference预编译库](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#c)页面根据开发环境选择对应的Jetpack C++包下载，并解压。

所有编译选项通过环境变量导入

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export BUILD_ON_JETSON=ON
export ENABLE_VISION=ON

# ENABLE_PADDLE_BACKEND & PADDLEINFERENCE_DIRECTORY为可选项
export ENABLE_PADDLE_BACKEND=ON
export PADDLEINFERENCE_DIRECTORY=/Download/paddle_inference_jetson

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
