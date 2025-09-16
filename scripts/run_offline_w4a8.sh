# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
set -ex
rm -rf log
rm -f core*

export devices=0
export CUDA_VISIBLE_DEVICES=${devices}
model_path=${1:-"/PATH/MODEL_PATH"}
output_path=${2:-"/PATH/OUTPUT_MODEL"}
moe_quant_type=${3:-"w4a8"}
for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
unset ${name}
done
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=1
export TRAINER_INSTANCES_NUM=1
export TRAINER_INSTANCES=`hostname -i`
self_ip=`hostname -i`

python offline_w4a8.py \
       --model_name_or_path ${model_path} \
       --output_dir ${output_path} \
       --safe_serialization "True" \
       --moe_quant_type ${moe_quant_type}
