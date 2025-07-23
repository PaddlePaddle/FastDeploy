"""
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
"""

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
args = parser.parse_args()
ep_num = 4
model_dir = args.model_dir

scale_dicts = []

for i in range(ep_num):
    scale_dict = {}
    path = os.path.join(model_dir, f"cachekv_scales_{i}.json")
    with open(path, "r") as file:
        scale_c = json.load(file)
        for k, v in scale_c.items():
            scale_dict[k] = v
    scale_dicts.append(scale_dict)

new_dict = {}
for k in scale_dicts[0].keys():
    v_list = [scale_dicts[i][k] for i in range(ep_num)]
    v = np.concatenate(v_list, axis=1)
    v = v.tolist()
    new_dict[k] = v

res_file = os.path.join(model_dir, "cachekv_scales.json")

b = json.dumps(new_dict)
f = open(res_file, "w")
f.write(b)
f.close()
