[English](../../en/build_and_install/sophgo.md) | 简体中文
# SOPHGO 部署库编译

## SOPHGO 环境准备
SOPHGO支持linux下进行编译,系统为Debian/Ubuntu  
安装包由三个文件构成
- [sophon-driver\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/KWqbmEcKp)
- [sophon-libsophon\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/PlvlBXhWY)
- [sophon-libsophon-dev\_0.4.2\_$arch.deb](http://219.142.246.77:65000/sharing/zTErLlpS7)

其中“$arch”为当前机器的硬件架构，使用以下命令可以获取当前的服务器arch：
```shell
uname -m
```
通常x86_64 机器对应的硬件架构为amd64,arm64 机器对应的硬件架构为 arm64:  
```text
- sophon-driver_0.4.2_$arch.deb
- sophon-libsophon_0.4.2_$arch.deb
- sophon-libsophon-dev_0.4.2_$arch.deb  
```

其中:sophon-driver 包含了 PCIe 加速卡驱动;sophon-libsophon 包含了运行时环境(库文
件、工具等);sophon-libsophon-dev 包含了开发环境(头文件等)。如果只是在部署环境上安
装,则不需要安装 sophon-libsophon-dev。
可以通过如下步骤安装:
```shell
#安装依赖库,只需要执行一次:
sudo apt install dkms libncurses5
#安装 libsophon:
sudo dpkg -i sophon-*.deb
#在终端执行如下命令,或者登出再登入当前用户后即可使用 bm-smi 等命令:
source /etc/profile
```
安装位置为：
```text
/opt/sophon/
├── driver-0.4.2
├── libsophon-0.4.2
|    ├──bin
|    ├──data
|    ├──include
|    └──lib
└── libsophon-current->/opt/sophon/libsophon-0.4.2
```

## C++ SDK编译安装
搭建好编译环境之后，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy  
mkdir build && cd build

# CMake configuration with Ascend
cmake -DENABLE_SOPHGO_BACKEND=ON  \
      -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-sophgo \
      -DENABLE_VISION=ON \
      ..

# Build FastDeploy Ascend C++ SDK
make -j8
make install
```  
编译完成之后，会在当前的build目录下生成 fastdeploy-sophgo 目录，编译完成。

## Python FastDeploy 库编译
搭建好编译环境之后，编译命令如下：
```bash
# Download the latest source code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/python
export ENABLE_SOPHGO_BACKEND=ON
export ENABLE_VISION=ON

python setup.py build
python setup.py bdist_wheel

#编译完成后,请用户自行安装当前目录的dist文件夹内的whl包.
```
