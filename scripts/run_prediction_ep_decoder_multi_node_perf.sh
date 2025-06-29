# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


# export IP_LIST='10.95.244.83,10.95.244.82'
export IP_LIST='10.95.244.83,10.95.244.82,10.95.246.141,10.95.246.145'
# export IP_LIST='10.95.244.83,10.95.244.82,10.95.246.141,10.95.246.145,10.95.246.162,10.95.247.31,10.95.247.39,10.95.246.158'

export EP_DECODER_PERF_TEST=True
export USE_CACHE_KV_INT8=True
export MAX_SEQ_LEN=5000
export MAX_DEC_LEN=64

mpirun \
-x EP_DECODER_PERF_TEST -x USE_CACHE_KV_INT8 -x MAX_SEQ_LEN -x MAX_DEC_LEN \
--host $IP_LIST  \
bash run_prediction_ep_decoder.sh ${1} ${2} ${BATCH_SIZE:-92} ${USE_MICRO_BATCH:-"False"} $IP_LIST
