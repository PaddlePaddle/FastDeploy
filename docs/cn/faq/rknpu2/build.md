# FastDeploy RKNPU2引擎编译

## FastDeploy后端支持详情
FastDeploy当前在RK平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |


## 板端编译FastDeploy C++ SDK

RKNPU2暂时仅支持linux系统, 以下教程在RK3568(debian 10)、RK3588(debian 11) 环境下完成。

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

# 编译配置详情见README文件，这里只介绍关键的几个配置
# -DENABLE_ORT_BACKEND:     是否开启ONNX模型，默认关闭
# -DENABLE_RKNPU2_BACKEND:  是否开启RKNPU模型，默认关闭
# -RKNN2_TARGET_SOC:             编译SDK的板子型号，只能输入RK356X或者RK3588，注意区分大小写
cmake ..  -DENABLE_ORT_BACKEND=ON \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DRKNN2_TARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```

### 板端编译Python SDK

RKNPU2暂时仅支持linux系统, 以下教程在RK3568(debian 10)、RK3588(debian 11) 环境下完成。Python打包依赖`wheel`，编译前请先执行`pip install wheel`

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
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
