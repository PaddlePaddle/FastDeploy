#!/usr/bin/env bash
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARGS=`getopt -a -o w:n:h:hs:tv -l WITH_GPU:,docker_name:,http_proxy:,https_proxy:,trt_version: -- "$@"`

eval set -- "${ARGS}"
echo "parse start"

while true
do
        case "$1" in
        -w|--WITH_GPU)
                WITH_GPU="$2"
                shift;;
        -n|--docker_name)
                docker_name="$2"
                shift;;
        -h|--http_proxy)
                http_proxy="$2"
                shift;;
        -hs|--https_proxy)
                https_proxy="$2"
                shift;;
        -tv|--trt_version)
                trt_version="$2"
                shift;;
        --)
                shift
                break;;
        esac
shift
done

if [ -z $WITH_GPU ];then
    WITH_GPU="ON"
fi

if [ -z $docker_name ];then
    docker_name="build_fd"
fi

if [ $WITH_GPU == "ON" ]; then

if [ -z $trt_version ]; then
    # The optional value of trt_version: ["8.4.1.5", "8.5.2.2"]
    trt_version="8.5.2.2"
fi

if [ $trt_version == "8.5.2.2" ]
then
    cuda_version="11.8"
    cudnn_version="8.6"
else
    cuda_version="11.6"
    cudnn_version="8.4"
fi

echo "start build FD GPU library"

if [ ! -d "./cmake-3.18.6-Linux-x86_64/" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.tar.gz
    tar -zxvf cmake-3.18.6-Linux-x86_64.tar.gz
    rm -rf cmake-3.18.6-Linux-x86_64.tar.gz
fi

if [ ! -d "./TensorRT-${trt_version}/" ]; then
    wget https://fastdeploy.bj.bcebos.com/resource/TensorRT/TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
    tar -zxvf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
    rm -rf TensorRT-${trt_version}.Linux.x86_64-gnu.cuda-${cuda_version}.cudnn${cudnn_version}.tar.gz
fi

nvidia-docker run -i --rm --name ${docker_name} \
           -v`pwd`/..:/workspace/fastdeploy \
           -e "http_proxy=${http_proxy}" \
           -e "https_proxy=${https_proxy}" \
           -e "trt_version=${trt_version}"\
           nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04  \
           bash -c \
           'export https_proxy_tmp=${https_proxy}
            export http_proxy_tmp=${http_proxy}
            cd /workspace/fastdeploy/python;
            rm -rf .setuptools-cmake-build dist build fastdeploy/libs/third_libs;
            apt-get update;
            apt-get install -y --no-install-recommends patchelf python3-dev python3-pip rapidjson-dev git;
            unset http_proxy
            unset https_proxy
            ln -s /usr/bin/python3 /usr/bin/python;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            export WITH_GPU=ON;
            export ENABLE_TRT_BACKEND=OFF;
            export TRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-${trt_version}/;
            export ENABLE_ORT_BACKEND=OFF;
            export ENABLE_PADDLE_BACKEND=OFF;
            export ENABLE_OPENVINO_BACKEND=OFF;
            export ENABLE_VISION=ON;
            export ENABLE_TEXT=ON;
            python setup.py build;
            python setup.py bdist_wheel;
            cd /workspace/fastdeploy;
            rm -rf build; mkdir -p build;cd build;
            cmake .. -DENABLE_TRT_BACKEND=ON -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DWITH_GPU=ON -DTRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-${trt_version}/ -DENABLE_PADDLE_BACKEND=ON -DENABLE_ORT_BACKEND=ON -DENABLE_OPENVINO_BACKEND=ON -DENABLE_VISION=OFF -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE2ONNX=ON -DENABLE_TEXT=OFF -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install;
            cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            export https_proxy=${https_proxy_tmp}
            export http_proxy=${http_proxy_tmp}
            cmake .. -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r22.12 -DTRITON_CORE_REPO_TAG=r22.12 -DTRITON_BACKEND_REPO_TAG=r22.12;
            make -j`nproc`'

echo "build FD GPU library done"

else

echo "start build FD CPU library"

docker run -i --rm --name ${docker_name} \
           -v`pwd`/..:/workspace/fastdeploy \
           -e "http_proxy=${http_proxy}" \
           -e "https_proxy=${https_proxy}" \
           paddlepaddle/fastdeploy:21.10-cpu-only-buildbase \
           bash -c \
           'export https_proxy_tmp=${https_proxy}
            export http_proxy_tmp=${http_proxy}
            cd /workspace/fastdeploy/python;
            rm -rf .setuptools-cmake-build dist build fastdeploy/libs/third_libs;
            ln -s /usr/bin/python3 /usr/bin/python;
            export WITH_GPU=OFF;
            export ENABLE_ORT_BACKEND=OFF;
            export ENABLE_PADDLE_BACKEND=OFF;
            export ENABLE_OPENVINO_BACKEND=OFF;
            export ENABLE_VISION=ON;
            export ENABLE_TEXT=ON;
            unset http_proxy
            unset https_proxy
            python setup.py build;
            python setup.py bdist_wheel;
            cd /workspace/fastdeploy;
            rm -rf build; mkdir build; cd build;
            cmake .. -DENABLE_TRT_BACKEND=OFF -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DWITH_GPU=OFF -DENABLE_PADDLE_BACKEND=ON -DENABLE_ORT_BACKEND=ON -DENABLE_OPENVINO_BACKEND=ON -DENABLE_VISION=OFF -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE2ONNX=ON -DENABLE_TEXT=OFF -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install;
            cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            export https_proxy=${https_proxy_tmp}
            export http_proxy=${http_proxy_tmp}
            cmake .. -DTRITON_ENABLE_GPU=OFF -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10;
            make -j`nproc`'

echo "build FD CPU library done"

fi
