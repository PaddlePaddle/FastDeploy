中文 ｜ [English](../EN/compile_without_docker_centos-en.md)

# FastDeploy Serving CentOS编译教程

本教程介绍如何在CentOS环境中安装依赖项、编译FastDeploy Serving并打包，用户最终可将部署包安装到CentOS系统中，不需要依赖docker容器。

如果部署环境有sudo权限，则可在部署环境中直接编译和打包。如果没有sudo权限，无法使用yum安装，则可以在开发机中创建与部署环境一致的docker容器进行编译和打包，最终将部署包上传至部署环境进行部署和运行。

本教程为GPU版本的FastDeploy Serving编译教程，对于CPU版的编译，可自行根据本教程内容进行裁剪，主要包括：

- 不需要CUDA、TensorRT、datacenter-gpu-manager等依赖
- 编译tritonserver时，去掉--enable-gpu和--enable-gpu-metrics
- 编译FastDeploy Runtime时关闭GPU、TensorRT等GPU相关的选项

## 1. 环境

* CentOS Linux release 7.9.2009
* CUDA 11.2（与部署环境中的CUDA版本保持一致）
* Python 3.8（可使用conda环境）
* GCC 9.4.0

## 2. 编译GCC

可按照以下步骤编译GCC 9.4.0，make install后，可将/opt/gcc-9.4.0/目录打包备份，后续可重复利用。

```
wget http://gnu.mirror.constant.com/gcc/gcc-9.4.0/gcc-9.4.0.tar.gz
tar xvf gcc-9.4.0.tar.gz
cd gcc-9.4.0
mkdir build
cd build
../configure --enable-languages=c,c++ --disable-multilib --prefix=/opt/gcc-9.4.0/
make -j8
make install
```

## 3. 安装tritonserver编译所需的依赖库

用yum安装的依赖项：

```
yum install numactl-devel
yum install libarchive-devel
yum install re2-devel

wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
rpm -ivh libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
rpm -ivh libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
```

安装rapidjson：

```
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
git submodule update --init
mkdir build && cd build
CC=/opt/gcc-9.4.0/bin/gcc CXX=
/opt/gcc-9.4.0/bin/g++
cmake ..
make install
```

安装boost 1.70：

```
wget https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_70_0_rc2.tar.gz
tar xvf boost_1_70_0_rc2.tar.gz
cd boost_1_70_0
./bootstrap.sh --prefix=/opt/boost
./b2 install --prefix=/opt/boost --with=all
```

安装datacenter-gpu-manager（libdcgm）：

```
dnf config-manager \
    --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
dnf clean expire-cache
dnf install -y datacenter-gpu-manager
```

## 4. 编译tritonserver

```
cd /workspace
git clone https://github.com/triton-inference-server/server.git -b r21.10
cd server
mkdir -p build/tritonserver/install
CC=/opt/gcc-9.4.0/bin/gcc CXX=/opt/gcc-9.4.0/bin/g++ \
BOOST_LIBRARYDIR=/opt/boost/lib BOOST_INCLUDEDIR=/opt/boost/include \
python build.py \
     --build-dir `pwd`/build \
     --no-container-build \
     --backend=ensemble \
     --enable-gpu \
     --endpoint=grpc \
     --endpoint=http \
     --enable-stats \
     --enable-tracing \
     --enable-logging \
     --enable-stats \
     --enable-metrics \
     --enable-gpu-metrics \
     --cmake-dir `pwd`/build \
     --repo-tag=common:r21.10 \
     --repo-tag=core:r21.10 \
     --repo-tag=backend:r21.10 \
     --repo-tag=thirdparty:r21.10 \
     --backend=python:r21.10
```

## 5. 编译FastDeploy Runtime和Serving

编译FastDeploy Runtime，这里需要依赖TensorRT，并指定TensorRT路径，8.0+版本都可以，需要与CUDA版本匹配

```
cd /workspace/
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
CC=/opt/gcc-9.4.0/bin/gcc CXX=/opt/gcc-9.4.0/bin/g++ cmake .. \
  -DENABLE_TRT_BACKEND=ON \
  -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install \
  -DWITH_GPU=ON \
  -DTRT_DIRECTORY=/workspace/TensorRT-8.4.3.1 \
  -DENABLE_PADDLE_BACKEND=ON \
  -DENABLE_ORT_BACKEND=ON \
  -DENABLE_OPENVINO_BACKEND=ON \
  -DENABLE_VISION=ON \
  -DBUILD_FASTDEPLOY_PYTHON=OFF \
  -DENABLE_PADDLE2ONNX=ON \
  -DENABLE_TEXT=OFF \
  -DLIBRARY_NAME=fastdeploy_runtime
make -j8
make install
```

编译Serving：

```
cd /workspace/FastDeploy/serving/
mkdir build && cd build
CC=/opt/gcc-9.4.0/bin/gcc CXX=/opt/gcc-9.4.0/bin/g++ cmake .. \
  -DFASTDEPLOY_DIR=/workspace/FastDeploy/build/fastdeploy_install \
  -DTRITON_COMMON_REPO_TAG=r21.10 \
  -DTRITON_CORE_REPO_TAG=r21.10 \
  -DTRITON_BACKEND_REPO_TAG=r21.10
make -j8
```

## 6. 打包

将Serving运行所需的可执行文件、脚本、依赖库等，统一放置在一个目录下，并压缩为tar.gz包。

```
# 打包的文件将统一放置在/workspace/opt目录下
cd /workspace/
mkdir /workspace/opt

# triton server
mkdir -p opt/tritonserver
cp -r /workspace/server/build/tritonserver/install/* opt/tritonserver

# python backend
mkdir -p opt/tritonserver/backends/python
cp -r /workspace/server/build/python/install/backends/python opt/tritonserver/backends/

# fastdeploy backend
mkdir -p opt/tritonserver/backends/fastdeploy
cp /workspace/FastDeploy/serving/build/libtriton_fastdeploy.so opt/tritonserver/backends/fastdeploy/

# rename tritonserver to fastdeployserver
mv opt/tritonserver/bin/tritonserver opt/tritonserver/bin/fastdeployserver

# fastdeploy runtime
cp -r /workspace/FastDeploy/build/fastdeploy_install/ opt/fastdeploy/

# GCC
cp -r /opt/gcc-9.4.0/ opt/
```

对于一些yum安装的依赖库，如果部署环境没有，也需要一同打包，放入opt/third_libs下，包括：

* /lib64/libdcgm.so.3
* /lib64/libnuma.so.1
* /lib64/libre2.so.0
* /lib64/libb64.so.0
* /lib64/libarchive.so.13

最终的opt/目录结构如下，README.md和init.sh需要打包人员添加，其中的README.md需要说明安装包的使用方法等，init.sh负责设置FastDeploy Serving运行所需的环境变量

```
opt/
├── fastdeploy
├── gcc-9.4.0
├── init.sh
├── README.md
└── tritonserver
└── third_libs
```

init.sh示例：

```
CURRENT_DIR=$(dirname $(readlink -f "${BASH_SOURCE}"))
echo $CURRENT_DIR
source $CURRENT_DIR/fastdeploy/fastdeploy_init.sh
export PATH=$CURRENT_DIR/tritonserver/bin:$PATH
export LD_LIBRARY_PATH=$CURRENT_DIR/gcc-9.4.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENT_DIR/tritonserver/backends/python/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CURRENT_DIR/third_libs:$LD_LIBRARY_PATH
unset CURRENT_DIR
```
