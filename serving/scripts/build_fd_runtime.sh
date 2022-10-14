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
WITH_GPU=${1:-ON}

if [ $WITH_GPU == "ON" ]; then

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

docker run -it --rm --name build_fd_runtime \
           -v`pwd`/..:/workspace/fastdeploy \
           nvcr.io/nvidia/tritonserver:21.10-py3-min \
           bash -c \
           'cd /workspace/fastdeploy;
            rm -rf build; mkdir build; cd build;
            apt-get update;
            apt-get install -y --no-install-recommends python3-dev python3-pip;
            ln -s /usr/bin/python3 /usr/bin/python;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            cmake .. -DENABLE_TRT_BACKEND=ON -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3 -DWITH_GPU=ON -DTRT_DIRECTORY=/workspace/fastdeploy/serving/TensorRT-8.4.1.5/ -DENABLE_PADDLE_BACKEND=ON -DENABLE_ORT_BACKEND=ON -DENABLE_OPENVINO_BACKEND=ON -DENABLE_VISION=OFF -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE_FRONTEND=ON -DENABLE_TEXT=OFF -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install'

else

docker run -it --rm --name build_fd_runtime \
           -v`pwd`/..:/workspace/fastdeploy \
           paddlepaddle/fastdeploy:22.09-cpu-only-buildbase \
           bash -c \
           'cd /workspace/fastdeploy;
            rm -rf build; mkdir build; cd build;
            cmake .. -DENABLE_TRT_BACKEND=OFF -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-0.0.3 -DWITH_GPU=OFF -DENABLE_PADDLE_BACKEND=ON -DENABLE_ORT_BACKEND=ON -DENABLE_OPENVINO_BACKEND=ON -DENABLE_VISION=OFF -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE_FRONTEND=ON -DENABLE_TEXT=OFF -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install'

fi
