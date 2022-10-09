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

docker run -it --rm --name build_fd_vison \
           -v`pwd`:/workspace/fastdeploy \
           nvcr.io/nvidia/tritonserver:21.10-py3-min \
           bash -c \
           'cd /workspace/fastdeploy/python;
            rm -rf .setuptools-cmake-build dist;
            apt-get update;
            apt-get install -y --no-install-recommends patchelf python3-dev python3-pip;
            ln -s /usr/bin/python3 /usr/bin/python;
            export PATH=/workspace/fastdeploy/cmake-3.18.6-Linux-x86_64/bin:$PATH;
            export WITH_GPU=ON;
            export ENABLE_ORT_BACKEND=OFF;
            export ENABLE_VISION=ON;
            export ENABLE_TEXT=ON;
            python setup.py build;
            python setup.py bdist_wheel'
