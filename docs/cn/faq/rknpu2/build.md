[English](../../../en/faq/rknpu2/build.md) | 中文
# FastDeploy RKNPU2引擎编译

## FastDeploy后端支持详情

FastDeploy当前在`RK356X/RK3588`平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |

## 编译安装FastDeploy C++ SDK

由于不同开发板的性能以及资源均不相同，我们提供了板端编译以及交叉编译两种方式来编译FastDeploy。
您可以根据需求从中选择一种来进行。

### FastDeploy后端支持详情

FastDeploy当前在`RK356X/RK3588`平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |

### 板端编译FastDeploy C++ SDK

对于内存比较充足且编译工具链完整的开发版，我们推荐直接在板端执行编译。
以下教程在RK356X(debian10),RK3588(debian 11) 环境下测试通过。

你可以通过修改以下参数来实现自定义你的FastDeploy工具包。

| 选项                      | 说明                                                                        |
|:------------------------|:--------------------------------------------------------------------------|
| ENABLE_ORT_BACKEND      | 默认OFF, 是否编译集成ONNX Runtime后端(CPU/GPU上推荐打开)                                 |
| ENABLE_LITE_BACKEND     | 默认OFF，是否编译集成Paddle Lite后端(编译Android库时需要设置为ON)                             |
| ENABLE_RKNPU2_BACKEND   | 默认OFF，是否编译集成RKNPU2后端(RK3588/RK3568/RK3566上推荐打开)                           |
| ENABLE_VISION           | 默认OFF，是否编译集成视觉模型的部署模块                                                     |
| RKNN2_TARGET_SOC        | ENABLE_RKNPU2_BACKEND时才需要使用这个编译选项。无默认值, 可输入值为RK3588/RK356X, 必须填入，否则 将编译失败 |
| ORT_DIRECTORY           | 当开启ONNX Runtime后端时，用于指定用户本地的ONNX Runtime库路径；如果不指定，编译过程会自动下载ONNX Runtime库  |
| OPENCV_DIRECTORY        | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库              |

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DENABLE_ORT_BACKEND=ON \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0

# build if soc is RK3588
make -j8
# build if soc is RK356X
make -j4

make install
```

### 交叉编译FastDeploy C++ SDK

对于内存不够充足或者编译工具链不够完整的开发版，我们也提供了交叉编译的方式来帮助你完成FastDeploy编译。
以下的交叉编译过程在Ubuntu22.04下测试通过。

在开始交叉编译前，你需要按以下步骤配置安装环境，以确保板端不会出现glibc对应不上的错误。

```bash
sudo apt install cmake build-essential
wget https://bj.bcebos.com/paddle2onnx/libs/gcc-linaro-6.3.1-2017.tar.gz
tar -xzvf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.gz -C /path/to/save
```

你可以通过修改以下参数来实现自定义你的FastDeploy工具包。

| 选项                      | 说明                                                                        |
|:------------------------|:--------------------------------------------------------------------------|
| ENABLE_RKNPU2_BACKEND   | 默认OFF，是否编译集成RKNPU2后端(RK3588/RK3568/RK3566上推荐打开)                           |
| ENABLE_VISION           | 默认OFF，是否编译集成视觉模型的部署模块                                                     |
| RKNN2_TARGET_SOC        | ENABLE_RKNPU2_BACKEND时才需要使用这个编译选项。无默认值, 可输入值为RK3588/RK356X, 必须填入，否则 将编译失败 |
| OPENCV_DIRECTORY        | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库              |

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DCMAKE_C_COMPILER=/path/to/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/path/to/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_TOOLCHAIN_FILE=./../cmake/toolchain.cmake \
          -DTARGET_ABI=arm64 \
          -DENABLE_ORT_BACKEND=OFF \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK356X \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0
make -j8
make install
```

### 配置环境变量

为了方便大家配置环境变量，FastDeploy提供了一键配置环境变量的脚本，在运行程序前，你需要执行以下命令

```bash
# 临时配置
source PathToFastDeploySDK/fastdeploy_init.sh

# 永久配置
source PathToFastDeploySDK/fastdeploy_init.sh
sudo cp PathToFastDeploySDK/fastdeploy_libs.conf /etc/ld.so.conf.d/
sudo ldconfig
```

## 编译FastDeploy Python SDK

除了NPU，Rockchip的芯片还有其他的一些功能。
这些功能大部分都是需要C/C++进行编程，因此如果您使用到了这些模块，我们不推荐您使用Python SDK.
Python SDK的编译暂时仅支持板端编译, 以下教程在RK3568(debian 10)、RK3588(debian 11) 环境下完成。Python打包依赖`wheel`，编译前请先执行`pip install wheel`


```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

cd python
export ENABLE_ORT_BACKEND=ON
export ENABLE_RKNPU2_BACKEND=ON
export ENABLE_VISION=ON

# 请根据你的开发版的不同，选择RK3588和RK356X
export RKNN2_TARGET_SOC=RK3588

# 如果你的核心板的运行内存大于等于8G，我们建议您执行以下命令进行编译。
python3 setup.py build
# 值得注意的是，如果你的核心板的运行内存小于8G，我们建议您执行以下命令进行编译。
python3 setup.py build -j1

python3 setup.py bdist_wheel
cd dist
pip3 install fastdeploy_python-0.0.0-cp39-cp39-linux_aarch64.whl
```
