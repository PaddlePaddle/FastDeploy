
# How to Build Jetson Deployment Library

FastDeploy currently supports only ONNX Runtime CPU and TensorRT GPU backend inference in Jetson

## How to Build and Install C++ SDK

Prerequisite for Compiling on NVIDIA Jetson:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.10.0
- jetpack >= 4.6.1

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DBUILD_ON_JETSON=ON \
         -DENABLE_VISION=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/installed_fastdeploy
make -j8
make install
```

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

## How to Build and Install Python SDK

Prerequisite for Compiling on NVIDIA Jetson:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.10.0
- jetpack >= 4.6.1
- python >= 3.6

All compilation options are imported via environment variables

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export BUILD_ON_JETSON=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel
```

The compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory once finished. Users can pip-install it directly.

During the compilation, if developers want to change the compilation parameters, it is advisable to delete the `build` and `.setuptools-cmake-build` subdirectories in the `FastDeploy/python` to avoid the possible impact from cache, and then recompile.
