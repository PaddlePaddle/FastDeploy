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

docker run -it --rm --name build_fd_backend \
           -v`pwd`/..:/workspace/fastdeploy \
           nvcr.io/nvidia/tritonserver:21.10-py3 \
           bash -c \
           'cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            apt-get update; apt-get install -y --no-install-recommends rapidjson-dev;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            cmake .. -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy-0.0.3 -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10; make -j`nproc`'
else
docker run -it --rm --name build_fd_backend \
           -v`pwd`/..:/workspace/fastdeploy \
           paddlepaddle/fastdeploy:22.09-cpu-only-buildbase \
           bash -c \
           'cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            apt-get update; apt-get install -y --no-install-recommends rapidjson-dev;
            cmake .. -DTRITON_ENABLE_GPU=OFF -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy-0.0.3 -DTRITON_COMMON_REPO_TAG=r22.09 -DTRITON_CORE_REPO_TAG=r22.09 -DTRITON_BACKEND_REPO_TAG=r22.09; make -j`nproc`'
fi
