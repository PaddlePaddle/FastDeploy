# Jetson部署环境

FastDeploy当前在Jetson仅支持ONNX Runtime CPU和TensorRT GPU两种后端推理

## C++ SDK编译安装

编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- jetpack >= 4.6

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
         -DENABLE_TRT_BACKEND=ON \
         -DWITH_GPU=ON \
         -DBUILD_ON_JETSON=ON \ 
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON
make -j8
make install
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库


## Python包编译安装

编译过程同样需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.18.0
- jetpack >= 4.6
- python >= 3.6

所有编译选项通过环境变量导入

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_ORT_BACKEND=ON
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export ENABLE_VISION=ON
export BUILD_ON_JETSON=ON

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
