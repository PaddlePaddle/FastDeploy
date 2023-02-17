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

ARG http_proxy
ARG https_proxy

FROM nvcr.io/nvidia/tritonserver:21.10-py3 as full
FROM nvcr.io/nvidia/tritonserver:21.10-py3-min

COPY --from=full /opt/tritonserver/bin/tritonserver /opt/tritonserver/bin/fastdeployserver
COPY --from=full /opt/tritonserver/lib /opt/tritonserver/lib
COPY --from=full /opt/tritonserver/include /opt/tritonserver/include
COPY --from=full /opt/tritonserver/backends/python /opt/tritonserver/backends/python

COPY serving/TensorRT-8.4.1.5 /opt/TensorRT-8.4.1.5

ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    DCGM_VERSION=2.2.9 \
    http_proxy=$http_proxy \
    https_proxy=$http_proxy

RUN apt-get update \
    && apt-key del 7fa2af80 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y --no-install-recommends datacenter-gpu-manager=1:2.2.9

RUN apt-get update \
    && apt-get install -y --no-install-recommends libre2-5 libb64-0d python3 python3-pip libarchive-dev ffmpeg libsm6 libxext6 \
    && python3 -m pip install -U pip \
    && python3 -m pip install paddlenlp fast-tokenizer-python

COPY python/dist/*.whl /opt/fastdeploy/
RUN python3 -m pip install  /opt/fastdeploy/*.whl \
    && rm -rf /opt/fastdeploy/*.whl

# unset proxy
ENV http_proxy=
ENV https_proxy=
RUN python3 -m pip install paddlepaddle-gpu==2.4.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

COPY serving/build/libtriton_fastdeploy.so /opt/tritonserver/backends/fastdeploy/
COPY build/fastdeploy_install /opt/fastdeploy/

ENV LD_LIBRARY_PATH="/opt/TensorRT-8.4.1.5/lib/:/opt/fastdeploy/lib:/opt/fastdeploy/third_libs/install/onnxruntime/lib:/opt/fastdeploy/third_libs/install/paddle2onnx/lib:/opt/fastdeploy/third_libs/install/tensorrt/lib:/opt/fastdeploy/third_libs/install/paddle_inference/paddle/lib:/opt/fastdeploy/third_libs/install/paddle_inference/third_party/install/mkldnn/lib:/opt/fastdeploy/third_libs/install/paddle_inference/third_party/install/mklml/lib:/opt/fastdeploy/third_libs/install/openvino/runtime/lib:$LD_LIBRARY_PATH"
ENV PATH="/opt/tritonserver/bin:$PATH"
