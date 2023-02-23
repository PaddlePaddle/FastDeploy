English | [中文](../../cn/build_and_install/jetson.md)

# How to Build FastDeploy Library on Nvidia Jetson Platform

FastDeploy supports CPU inference with ONNX Runtime and GPU inference with Nvidia TensorRT/Paddle Inference on Nvidia Jetson platform

- If there's error occurs, shows `Could not find a package configuration file provided by "Python" with any of the following names: PythonConfig.cmake python-config.cmake`, please try to [upgrade cmake to 3.25 or newer version](https://cmake.org/download/) to solve the problem.
- 
## How to Build and Install FastDeploy C++ Library

Prerequisite for Compiling on NVIDIA Jetson:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.10.0
- jetpack >= 4.6.1

If you need to integrate Paddle Inference backend(Support CPU/GPU)，please download and decompress the prebuilt library in [Paddle Inference prebuild libraries](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#c) according to your develop envriment.

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DBUILD_ON_JETSON=ON \
         -DENABLE_VISION=ON \
         -DENABLE_PADDLE_BACKEND=ON \ # This is optional, can be OFF if you don't need
         -DPADDLEINFERENCE_DIRECTORY=/Download/paddle_inference_jetson \
         -DCMAKE_INSTALL_PREFIX=${PWD}/installed_fastdeploy
make -j8
make install
```

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

## How to Build and Install FastDeploy Python Library

Prerequisite for Compiling on NVIDIA Jetson:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.10.0
- jetpack >= 4.6.1
- python >= 3.6

Notice the `wheel` is required if you need to pack a wheel, execute `pip install wheel` first.

If you need to integrate Paddle Inference backend(Support CPU/GPU)，please download and decompress the prebuilt library in [Paddle Inference prebuild libraries](https://www.paddlepaddle.org.cn/inference/v2.4/guides/install/download_lib.html#c) according to your develop envriment.

All compilation options are imported via environment variables

```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export BUILD_ON_JETSON=ON
export ENABLE_VISION=ON

# ENABLE_PADDLE_BACKEND & PADDLEINFERENCE_DIRECTORY are optional
export ENABLE_PADDLE_BACKEND=ON
export PADDLEINFERENCE_DIRECTORY=/Download/paddle_inference_jetson

python setup.py build
python setup.py bdist_wheel
```

The compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory once finished. Users can pip-install it directly.

During the compilation, if developers want to change the compilation parameters, it is advisable to delete the `build` and `.setuptools-cmake-build` subdirectories in the `FastDeploy/python` to avoid the possible impact from cache, and then recompile.
