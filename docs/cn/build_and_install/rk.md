# RK 部署库编译和分类模型运行示例

FastDeploy在瑞芯微（Rockchip）Soc上支持基于芯原 TIM-VX 的 Paddle-Lite 后端推理。芯原（verisilicon）作为 IP 设计厂商，本身并不提供实体SoC产品，而是授权其 IP 给芯片厂商，如：晶晨（Amlogic），瑞芯微（Rockchip）等。因此本文是适用于被芯原授权了 NPU IP 的芯片产品。只要芯片产品没有大副修改芯原的底层库，则该芯片就可以使用本文档作为推理部署的参考和教程，更多详细的信息请参考：[PaddleLite芯原 TIM-VX 部署示例](https://paddle-lite.readthedocs.io/zh/develop/demo_guides/verisilicon_timvx.html)

相关编译选项说明如下：  
|编译选项|默认值|说明|备注|  
|:---|:---|:---|:---|  
|ENABLE_LITE_BACKEND|OFF|编译RK库时需要设置为ON| - |
|WITH_LITE_STATIC|OFF|是否使用Lite静态库| 暂不支持使用Lite静态库 |
|WITH_LITE_FULL_API|ON|是否使用Lite Full API库| 目前必须为ON |

更多编译选项请参考[FastDeploy编译选项说明](./README.md)

## RK C++ SDK 编译安装  

RK C++ SDK需要交叉编译后再部署到RK上，接下会介绍交叉编译环境的搭建以及C++ SDK的编译

### 交叉编译环境搭建

#### 宿主机环境需求  
- os：Ubuntu == 16.04
- cmake： version >= 3.10.0  

#### 环境搭建
```bash
 # 1. Install basic software
apt update
apt-get install -y --no-install-recommends \
  gcc g++ git make wget python unzip

# 2. Install arm gcc toolchains
apt-get install -y --no-install-recommends \
  g++-arm-linux-gnueabi gcc-arm-linux-gnueabi \
  g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 3. Install cmake 3.10 or above
wget -c https://mms-res.cdn.bcebos.com/cmake-3.10.3-Linux-x86_64.tar.gz && \
  tar xzf cmake-3.10.3-Linux-x86_64.tar.gz && \
  mv cmake-3.10.3-Linux-x86_64 /opt/cmake-3.10 && \
  ln -s /opt/cmake-3.10/bin/cmake /usr/bin/cmake && \
  ln -s /opt/cmake-3.10/bin/ccmake /usr/bin/ccmake
```

### SDK编译
搭建好交叉编译环境之后，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# Setting up RK toolchanin
TARGET_ABI=armhf
TOOLCHAIN_FILE=../cmake/rk.cmake
FASDEPLOY_INSTALL_DIR=./arm_install

# CMake configuration with RK toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DENABLE_PADDLE_FRONTEND=OFF \
      -DTARGET_ABI=${TARGET_ABI} \
      -DENABLE_LITE_BACKEND=ON \
      -DENABLE_VISION=ON \
      -DENABLE_VISION_VISUALIZE=ON \
      -DBUILD_EXAMPLES=ON \
      -DCMAKE_INSTALL_PREFIX=${FASDEPLOY_INSTALL_DIR} \
      -Wno-dev ..

# Build FastDeploy RK C++ SDK
make -j8
make install  
```  
编译完成后，RK C++ SDK 保存在 `build/arm_install` 目录下。

## RK 分类模型运行示例  
在 FastDeploy 编译完成之后，接下来我们会介绍如何在 Rockchip RV1126 开发板上使用 adb 工具运行一个分类模型，部署命令如下：
```bash
cd FastDeploy/tools/rk_deploy
# copy_libs.sh 会将所有需要用的库和可执行文件都拷贝到当前目录下，传入的vision_classification_paddleclas_infer 是需要运行的可执行文件，你也可以根据需要换成你想要执行的文件，所有可执行文件都在 FastDeploy/build/bin下，运行成功后当前目录下会新增 libs 和 vision_classification_paddleclas_infer 两个文件
bash copy_libs.sh vision_classification_paddleclas_infer

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 基于 adb 部署模型到 Rockchip RV1126，DEVICE_ID 可用 adb devices 查询
bash run_with_adb.sh vision_classification_paddleclas_infer ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg $DEVICE_ID
```
部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/198015942-b5f27cea-e62e-4efe-9248-085e4f468e0f.jpg">

需要特别注意的是，在RK上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](./../quantize.md)
