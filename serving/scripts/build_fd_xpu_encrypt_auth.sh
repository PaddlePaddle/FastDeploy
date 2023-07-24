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


echo "start build FD XPU AUTH library"

docker run -i --rm --name build_fd_xpu_auth_108_dev \
           -v `pwd`/..:/workspace/fastdeploy \
           -e "http_proxy=${http_proxy}" \
           -e "https_proxy=${https_proxy}" \
           -e "no_proxy=${no_proxy}" \
           -e "PADDLEINFERENCE_URL=${PADDLEINFERENCE_URL}" \
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
            wget -q ${PADDLEINFERENCE_URL} && tar -zxvf ${PADDLEINFERENCE_URL##*/};
            tmp_dir=${PADDLEINFERENCE_URL##*/}
            mv ${tmp_dir%.*} paddle_inference
            PADDLEINFERENCE_DIRECTORY=${PWD}/paddle_inference
            rm -rf build; mkdir build; cd build;
            cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy_install -DWITH_KUNLUNXIN=ON -DENABLE_PADDLE_BACKEND=ON -DPADDLEINFERENCE_DIRECTORY=${PADDLEINFERENCE_DIRECTORY} -DENABLE_BENCHMARK=ON -DLIBRARY_NAME=fastdeploy_runtime;
            make -j`nproc`;
            make install;
            # fix the link error of libbkcl.so
            mkdir -p /home/users/yanzikui/wenxin/baidu/xpu/bkcl/output/so;
            cp /workspace/fastdeploy/build/fastdeploy_install/third_libs/install/paddle_inference/third_party/install/xpu/lib/libbkcl.so /home/users/yanzikui/wenxin/baidu/xpu/bkcl/output/so;
            cd /workspace/fastdeploy/serving;
            rm -rf build; mkdir build; cd build;
            export https_proxy=${https_proxy_tmp}
            export http_proxy=${http_proxy_tmp}
            cmake .. -DTRITON_ENABLE_GPU=OFF -DFASTDEPLOY_DIR=/workspace/fastdeploy/build/fastdeploy_install -DTRITON_COMMON_REPO_TAG=r21.10 -DTRITON_CORE_REPO_TAG=r21.10 -DTRITON_BACKEND_REPO_TAG=r21.10;
            make -j`nproc`;
            echo $PADDLEINFERENCE_URL;
            '

echo "build FD XPU AUTH library done"
