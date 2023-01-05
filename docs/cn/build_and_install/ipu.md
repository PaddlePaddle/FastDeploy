[English](../../en/build_and_install/ipu.md) | 简体中文

# IPU部署库编译

## 编译选项说明

无论是在何平台编译，编译时仅根据需求修改如下选项，勿修改其它参数
| 选项                      | 支持平台 | 说明                                                                        |
|:------------------------|:------- | :--------------------------------------------------------------------------|
| WITH_IPU | Linux(x64) | 默认OFF，当编译支持IPU时，需设置为ON |
| ENABLE_ORT_BACKEND      | Linux(x64) | 默认OFF, 是否编译集成ONNX Runtime后端(仅支持CPU)    |
| ENABLE_PADDLE_BACKEND   | Linux(x64) | 默认OFF，是否编译集成Paddle Inference后端(支持IPU & CPU)                         |  
| ENABLE_OPENVINO_BACKEND | Linux(x64) | 默认OFF，是否编译集成OpenVINO后端(仅支持CPU)       |
| ENABLE_VISION           | Linux(x64) |  默认OFF，是否编译集成视觉模型的部署模块                                                    |
| ENABLE_TEXT             | Linux(x64) | 默认OFF，是否编译集成文本NLP模型的部署模块                                                  |

第三方库依赖指定（不设定如下参数，会自动下载预编译库）
| 选项                     | 说明                                                                                           |
| :---------------------- | :--------------------------------------------------------------------------------------------- |
| ORT_DIRECTORY           | 当开启ONNX Runtime后端时，用于指定用户本地的ONNX Runtime库路径；如果不指定，编译过程会自动下载ONNX Runtime库  |
| OPENCV_DIRECTORY        | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库              |
| OPENVINO_DIRECTORY      | 当开启OpenVINO后端时, 用于指定用户本地的OpenVINO库路径；如果不指定，编译过程会自动下载OpenVINO库             |

## C++ SDK编译安装

Linux编译需满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_PADDLE_BACKEND=ON \
         -DWITH_IPU=ON \
         -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
         -DENABLE_VISION=ON \
         -DENABLE_TEXT=ON
make -j8
make install
```

编译完成后，即在`CMAKE_INSTALL_PREFIX`指定的目录下生成C++推理库


## Python编译安装

Linux编译过程同样需要满足
- gcc/g++ >= 5.4(推荐8.2)
- cmake >= 3.16.0, < 3.23.0
- popart >= 3.0.0
- python >= 3.6

Python打包依赖`wheel`，编译前请先执行`pip install wheel`

所有编译选项通过环境变量导入

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_VISION=ON
export ENABLE_TEXT=ON
export ENABLE_PADDLE_BACKEND=ON
export WITH_IPU=ON

python setup.py build
python setup.py bdist_wheel
```

编译完成即会在`FastDeploy/python/dist`目录下生成编译后的`wheel`包，直接pip install即可

编译过程中，如若修改编译参数，为避免带来缓存影响，可删除`FastDeploy/python`目录下的`build`和`.setuptools-cmake-build`两个子目录后再重新编译
