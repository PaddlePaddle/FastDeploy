# RK2代NPU部署库编译

## 写在前面
FastDeploy已经初步支持RKNPU2的部署，目前暂时仅支持c++部署。使用的过程中，如果出现Bug请提Issues反馈。

## 简介
FastDeploy当前在RK平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |


## C++ SDK编译安装

RKNPU2仅支持linux下进行编译,以下教程均在linux环境下完成。

### 更新驱动和安装编译时需要的环境


在运行代码之前，我们需要安装以下最新的RKNPU驱动，目前驱动更新至1.4.0。为了简化安装我编写了快速安装脚本，一键即可进行安装。

**方法1: 通过脚本安装**
```bash
# 下载解压rknpu2_device_install_1.4.0
链接:https://pan.baidu.com/s/1yNww64gQnvwiCfNhELtkwQ?pwd=easy 提取码:easy 复制这段内容后打开百度网盘手机App，操作更方便哦

cd rknpu2_device_install_1.4.0
# RK3588运行以下代码
sudo rknn_install_rk3588.sh
# RK356X运行以下代码
sudo rknn_install_rk356X.sh
```

**方法2: 通过gittee安装**
```bash
# 安装必备的包
sudo apt update -y
sudo apt install -y python3 
sudo apt install -y python3-dev 
sudo apt install -y python3-pip 
sudo apt install -y gcc
sudo apt install -y python3-opencv
sudo apt install -y python3-numpy
sudo apt install -y cmake

# 下载rknpu2
# RK3588运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

# RK356X运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
```

### 编译C++ SDK

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

# 编译配置详情见README文件，这里只介绍关键的几个配置
# -DENABLE_ORT_BACKEND:     是否开启ONNX模型，默认关闭
# -DENABLE_RKNPU2_BACKEND:  是否开启RKNPU模型，默认关闭
# -DTARGET_SOC:             编译SDK的板子型号，只能输入RK356X或者RK3588，注意区分大小写
cmake ..  -DENABLE_ORT_BACKEND=ON \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DTARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3
make -j8
make install
```

## 部署模型

请查看[RKNPU2部署模型教程](~/examples/rknpu2/README.md)