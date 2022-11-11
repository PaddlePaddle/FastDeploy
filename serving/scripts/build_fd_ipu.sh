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

# build vision
docker run -it --rm --name build_fd_vison \
           -v`pwd`/..:/workspace/fastdeploy \
           graphcore/poplar:3.0.0 \
           bash -c \
           'cd /workspace/fastdeploy/python;
            rm -rf .setuptools-cmake-build dist;
            apt-get update;
            apt-get install -y --no-install-recommends patchelf python3-dev python3-pip python3-setuptools build-essential;
            ln -s /usr/bin/python3 /usr/bin/python;
            pip3 install wheel;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            export WITH_GPU=OFF;
            export WITH_IPU=ON;
            export ENABLE_PADDLE_BACKEND=ON;
            export ENABLE_VISION=ON;
            python setup.py build;
            python setup.py bdist_wheel'

# build runtime
docker run -it --rm --name build_fd_runtime \
           -v`pwd`/..:/workspace/fastdeploy \
           graphcore/poplar:3.0.0 \
           bash -c \
           "cd /workspace/fastdeploy;
            rm -rf build; mkdir build; cd build;
            apt-get update;
            apt-get install -y --no-install-recommends python3-dev python3-pip build-essential;
            ln -s /usr/bin/python3 /usr/bin/python;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            cmake .. -DENABLE_ORT_BACKEND=OFF -DENABLE_TEXT=OFF -DENABLE_VISION=OFF -DBUILD_FASTDEPLOY_PYTHON=OFF -DENABLE_PADDLE_BACKEND=ON -DWITH_IPU=ON -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install"

# build backend
docker run -it --rm --name build_fd_backend \
           -v`pwd`/..:/workspace/fastdeploy \
           graphcore/poplar:3.0.0 \
           bash -c \
           "cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            apt-get update; apt-get install -y --no-install-recommends rapidjson-dev build-essential git ca-certificates;
            export PATH=/workspace/fastdeploy/serving/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            cmake .. -DTRITON_ENABLE_GPU=OFF -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10; make -j`nproc`"
