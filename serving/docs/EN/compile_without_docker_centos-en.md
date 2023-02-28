English | [中文](../zh_CN/compile_without_docker_centos.md)

# FastDeploy Serving CentOS Compilation Tutorial

This tutorial introduces how to install dependencies, compile and package FastDeploy Serving in the CentOS environment, and the user can finally install the deployment package into the CentOS system without relying on the docker container.

If the deployment environment has `sudo` permission, it can be compiled and packaged directly in the deployment environment. If you do not have `sudo` permission and cannot use `yum` to install, you can create a docker container which has the same environment as the deployment machine to compile and package, and finally upload the package to the deployment environment.

This tutorial is for GPU environment. For CPU-Only enviroment, you can tailor it according to the content of this tutorial, mainly including:

- No need for CUDA, TensorRT, datacenter-gpu-manager and other GPU dependencies
- When compiling tritonserver, remove --enable-gpu and --enable-gpu-metrics
- Disable GPU-related options such as WITH_GPU and ENABLE_TRT_BACKEND when compiling FastDeploy Runtime

## 1. Environments

* CentOS Linux release 7.9.2009
* CUDA 11.2 (consistent with the deployment env)
* Python 3.8 (prefer to use conda)
* GCC 9.4.0

## 2. Compile GCC

Follow the steps below to compile GCC 9.4.0. After `make install`, you can package the /opt/gcc-9.4.0/ directory for backup, which can be reused later.

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

## 3. Install dependencies for tritonserver

Dependencies which can be installed by `yum`:

```
yum install numactl-devel
yum install libarchive-devel
yum install re2-devel

wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
wget http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
rpm -ivh libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm
rpm -ivh libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm
```

Install rapidjson:

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

Install boost 1.70:

```
wget https://boostorg.jfrog.io/artifactory/main/release/1.70.0/source/boost_1_70_0_rc2.tar.gz
tar xvf boost_1_70_0_rc2.tar.gz
cd boost_1_70_0
./bootstrap.sh --prefix=/opt/boost
./b2 install --prefix=/opt/boost --with=all
```

Install datacenter-gpu-manager:

```
dnf config-manager \
    --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
dnf clean expire-cache
dnf install -y datacenter-gpu-manager
```

## 4. Compile tritonserver

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

## 5. Compile FastDeploy Runtime and Serving

FastDeploy Runtime depends on TensorRT for GPU serving, so TRT_DIRECTORY is required.

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

Compile FastDeploy Serving:

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

## 6. Package

Put the executable files, scripts, dependent libraries, etc. required for Serving to run inti one directory, and compress it into a tar.gz package.

```
# Put everything under /workspace/opt/
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

For some dependent libraries installed by yum, if the deployment environment does not have them, they also need to be packaged together and placed under opt/third_libs, including:

* /lib64/libdcgm.so.3
* /lib64/libnuma.so.1
* /lib64/libre2.so.0
* /lib64/libb64.so.0
* /lib64/libarchive.so.13

The final opt/ directory structure is as follows. README.md and init.sh need to be added by the packager. README.md needs to explain how to use the installation package, etc. init.sh is responsible for setting the environment variables required for FastDeploy Serving to run.

```
opt/
├── fastdeploy
├── gcc-9.4.0
├── init.sh
├── README.md
└── tritonserver
└── third_libs
```

init.sh example:

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
