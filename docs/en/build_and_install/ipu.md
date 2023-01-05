English | [中文](../../cn/build_and_install/ipu.md)

# How to Build IPU Deployment Environment

## Build Options

Please do not modify other cmake paramters exclude the following options.

| Option                      | Supported Platform | Description                                                                        |
|:------------------------|:------- | :--------------------------------------------------------------------------|
| ENABLE_ORT_BACKEND      | Linux(x64) | Default OFF, whether to intergrate ONNX Runtime backend(Only support CPU)   |
| ENABLE_PADDLE_BACKEND   | Linux(x64) | Default OFF, whether to intergrate Paddle Inference backend(Support IPU & CPU)             |               
| ENABLE_OPENVINO_BACKEND | Linux(x64) | Default OFF, whether to intergrate OpenVINO backend(Only support CPU)      |
| ENABLE_VISION           | Linux(x64) | Default OFF, whether to intergrate vision models |
| ENABLE_TEXT             | Linux(x64) | Default OFF, whether to intergrate text models |

The configuration for third libraries(Optional, if the following option is not defined, the prebuilt third libraries will download automaticly while building FastDeploy).
| Option                     | Description                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| ORT_DIRECTORY           | While ENABLE_ORT_BACKEND=ON, use ORT_DIRECTORY to specify your own ONNX Runtime library path.  |
| OPENCV_DIRECTORY        | While ENABLE_VISION=ON, use OPENCV_DIRECTORY to specify your own OpenCV library path.     |
| OPENVINO_DIRECTORY      |  While ENABLE_OPENVINO_BACKEND=ON, use OPENVINO_DIRECTORY to specify your own OpenVINO library path.    |

## How to Build and Install C++ SDK

Prerequisite for Compiling on Linux:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_PADDLE_BACKEND=ON \
         -DWITH_IPU=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON
make -j8
make install
```

Once compiled, the C++ inference library is generated in the directory specified by `CMAKE_INSTALL_PREFIX`

## How to Build and Install Python SDK

Prerequisite for Compiling on Linux:

- gcc/g++ >= 5.4 (8.2 is recommended)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0
- python >= 3.6

Notice the `wheel` is required if you need to pack a wheel, execute `pip install wheel` first.

All compilation options are imported via environment variables

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_VISION=ON
export ENABLE_PADDLE_BACKEND=ON
export WITH_IPU=ON

python setup.py build
python setup.py bdist_wheel
```

The compiled `wheel` package will be generated in the `FastDeploy/python/dist` directory once finished. Users can pip-install it directly.

During the compilation, if developers want to change the compilation parameters, it is advisable to delete the `build` and `.setuptools-cmake-build` subdirectories in the `FastDeploy/python` to avoid the possible impact from cache, and then recompile.
