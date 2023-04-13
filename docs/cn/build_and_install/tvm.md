[English]() | 简体中文

# TVM后端编译文档

ApacheTVM是一个用于CPU、GPU和机器学习加速器的开源机器学习编译器框架。它旨在使机器学习工程师能够在任何硬件后端高效地优化和运行计算。

## 编译选项说明

| 选项                 | 说明                                                           |
|:-------------------|:-------------------------------------------------------------|
| ENABLE_TVM_BACKEND | 默认OFF，是否编译集成TVM后端                                            |
| ENABLE_VISION      | 默认OFF，是否编译集成视觉模型的部署模块                                        |
| OPENCV_DIRECTORY   | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库 |

```bash
mkdir build
cd build
cmake ..  -DENABLE_TVM_BACKEND=ON \
	      -DENABLE_VISION=ON \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-tvm-0.0.0
make -j8
make install
```

## 编译自己的TVM库

如果你需要编译你自己的TVM库，请参考[TVM官方编译文档](https://tvm.apache.org/docs/install/from_source.html#build-the-shared-library)编译，这里仅给出简单的介绍。

### Mac平台编译TVM库

在安装TVM前你需要安装一些前置的依赖包，例如你可以使用brew进行如下依赖包的安装。

```bash
brew install gcc git cmake
brew install llvm
brew install miniforge
```

安装完依赖包后，你需要创建一个python环境来安装TVM(Python 版本的TVM将被用于转换模型)

```bash
conda init
conda create --name tvm python=3.8
```

创建Python环境后，你需要下载TVM，并编译他的配置项

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git submodule init
git submodule update
mkdir build
cp cmake/config.cmake build
```

例如我这里对LLVM的路径进行了编辑，你可以输入`brew list llvm`来找到`llvm-config`的具体位置。
```bash
vim build/config.cmake

# in line 145
set(USE_LLVM OFF) -> set(USE_LLVM /path/to/your/llvm-config)
```

随后即可进行编译，例如你可以输入以下命令来生成你自己的tvm库

```bash
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${PWD}/tvm
make -j8
make install
```

由于TVM对第三库的头文件存在依赖且打包时没有把这部分打包进去，因此我们需要手动添加第三方库的头文件，例如你可以输入以下命令

```bash
cd ..
cp -r 3rdparty/dmlc-core/include/dmlc ./build/tvm/include
cp -r 3rdparty/dlpack/include/dlpack ./build/tvm/include
```

随后我们需要把TVM安装到Python环境中，你可以参考以下命令进行安装

```bash
# Python dependencies
pip3 install --user numpy decorator attrs
pip3 install --user typing-extensions psutil scipy
pip3 install --user tornado
pip3 install --user tornado psutil 'xgboost>=1.1.0' cloudpickle
brew install openblas gfortran
pip install pybind11 cython pythran
pip install scipy --no-use-pep517
# Install TVM
export MACOSX_DEPLOYMENT_TARGET=10.9  # This is required for mac to avoid symbol conflicts with libstdc++
cd python;
python setup.py install --user;
```
