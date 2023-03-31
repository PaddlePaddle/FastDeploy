[English](../../../en/faq/rknpu2/build.md) | 中文
# FastDeploy RKNPU2引擎编译

## FastDeploy后端支持详情

FastDeploy当前在`RK356X/RK3588`平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |

## 编译FastDeploy SDK

针对RK356X和RK3588的性能差异，我们提供了两种编译FastDeploy的方式。

### 板端编译FastDeploy C++ SDK

针对RK3588，其CPU性能较强，板端编译的速度还是可以接受的，我们推荐在板端上进行编译。以下教程在RK356X(debian10),RK3588(debian 11) 环境下完成。

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
make -j8
make install
```

### 交叉编译FastDeploy C++ SDK

针对RK356X，其CPU性能较弱，我们推荐使用交叉编译进行编译。以下教程在Ubuntu 22.04环境下完成。

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# 如果您使用的是develop分支输入以下命令
git checkout develop

mkdir build && cd build
cmake ..  -DCMAKE_C_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ \
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

如果你找不到编译工具，你可以复制[交叉编译工具](https://bj.bcebos.com/paddle2onnx/libs/gcc-linaro-6.3.1-2017.zip)进行下载。

### 板端编译Python SDK

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
