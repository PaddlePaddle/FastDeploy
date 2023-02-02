English | [中文](../../../cn/faq/rknpu2/build.md) 
# FastDeploy RKNPU2 Engine Compilation 

## FastDeploy supported backends
FastDeploy currently supports the following backends on the RK platform: 

| Backend                | Platform                    | Supported model formats  | Notes                                          |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | Compile switch `ENABLE_ORT_BACKEND` is controlled by ON or OFF. Default OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | Compile switch `ENABLE_RKNPU2_BACKEND` is controlled by ON or OFF. Default OFF  |

## Compile FastDeploy SDK

### Compile FastDeploy C++ SDK on board side 

Currently, RKNPU2 is only available on linux. The following tutorial is completed on RK3568(debian 10) and RK3588(debian 11). 

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# If you are using the develop branch, type the following command 
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

### Cross-compile FastDeploy C++ SDK
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# If you are using the develop branch, type the following command 
git checkout develop

mkdir build && cd build
cmake ..  -DCMAKE_C_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=/home/zbc/opt/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ \
          -DCMAKE_TOOLCHAIN_FILE=./../cmake/toolchain.cmake \
          -DTARGET_ABI=arm64 \
          -DENABLE_ORT_BACKEND=OFF \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK3588 \
	      -DENABLE_FLYCV=ON \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.0
make -j8
make install
```

### Compile the Python SDK on the board

Currently, RKNPU2 is only available on linux. The following tutorial is  completed on RK3568(debian 10) and RK3588(debian 11). Packing Python is dependent on `wheel`, so run `pip install wheel` before compiling.

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy

# If you are using the develop branch, type the following command 
git checkout develop

cd python
export ENABLE_ORT_BACKEND=ON
export ENABLE_RKNPU2_BACKEND=ON
export ENABLE_VISION=ON
export RKNN2_TARGET_SOC=RK3588
python3 setup.py build
python3 setup.py bdist_wheel
cd dist
pip3 install fastdeploy_python-0.0.0-cp39-cp39-linux_aarch64.whl
```
