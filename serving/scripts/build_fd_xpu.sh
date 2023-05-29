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


echo "start build FD XPU library"

docker run -i --rm --name build_fd_xpu \
           -v `pwd`/..:/workspace/fastdeploy \
           -e "http_proxy=${http_proxy}" \
           -e "https_proxy=${https_proxy}" \
           -e "no_proxy=${no_proxy}" \
           --network=host --privileged \
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
            cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DWITH_KUNLUNXIN=ON -DENABLE_PADDLE_BACKEND=ON -DENABLE_VISION=ON -DENABLE_BENCHMARK=ON -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install;
            cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            export https_proxy=${https_proxy_tmp}
            export http_proxy=${http_proxy_tmp}
            cmake .. -DTRITON_ENABLE_GPU=OFF -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10;
            make -j`nproc`;
            cd /workspace/fastdeploy/benchmark/cpp;
            rm -rf build; mkdir build; cd build;
            unset http_proxy
            unset https_proxy
            cmake .. -DFASTDEPLOY_INSTALL_DIR=/workspace/fastdeploy/build/fastdeploy_install;
            make -j`nproc`;
            wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_infer.tgz && tar -zxvf ResNet50_infer.tgz;
            wget https://bj.bcebos.com/paddlehub/fastdeploy/000000014439.jpg;
            rm -f ResNet50_infer.tgz;
            rm -rf CMakeFiles;
            '

echo "build FD XPU library done"
