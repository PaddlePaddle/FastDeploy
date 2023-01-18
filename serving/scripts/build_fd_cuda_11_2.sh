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

if [ ! -d "./cmake-3.18.6-Linux-x86_64/" ]; then
    wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.tar.gz
    tar -zxvf cmake-3.18.6-Linux-x86_64.tar.gz
    rm -rf cmake-3.18.6-Linux-x86_64.tar.gz
fi

if [ ! -d "./TensorRT-8.4.1.5/" ]; then
    wget https://fastdeploy.bj.bcebos.com/third_libs/TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
    tar -zxvf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
    rm -rf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
fi

# build vision、runtime、backend
docker run -it --rm --name build_fd_libs \
           -v`pwd`/..:/workspace/fastdeploy \
           -e "http_proxy=${http_proxy}" \
           -e "https_proxy=${https_proxy}" \
           nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04 \
           bash -c \
           'cd /workspace/fastdeploy/python;
            rm -rf .setuptools-cmake-build dist build fastdeploy/libs/third_libs;
            apt-get update;
            apt-get install -y --no-install-recommends patchelf python3-dev python3-pip rapidjson-dev git;
            ln -s /usr/bin/python3 /usr/bin/python;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            export WITH_GPU=ON;
            export ENABLE_TRT_BACKEND=OFF;
            export TRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-8.4.1.5/;
            export ENABLE_ORT_BACKEND=OFF;
            export ENABLE_PADDLE_BACKEND=OFF;
            export ENABLE_OPENVINO_BACKEND=OFF;
            export ENABLE_VISION=ON;
            export ENABLE_TEXT=ON;
            python setup.py build;
            python setup.py bdist_wheel;
            cd /workspace/fastdeploy;
            rm -rf build; mkdir -p build;cd build;
            cmake .. -DENABLE_TRT_BACKEND=ON -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DWITH_GPU=ON -DTRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-8.4.1.5/ -DENABLE_PADDLE_BACKEND=ON -DENABLE_ORT_BACKEND=ON -DENABLE_OPENVINO_BACKEND=ON -DENABLE_VISION=ON -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE2ONNX=ON -DENABLE_TEXT=OFF -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install;
            cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            cmake .. -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10;
            make -j`nproc`'
