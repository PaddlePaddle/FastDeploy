# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
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
test resnet50
"""

import time
import poros
import torch
from torchvision import models

torch.set_grad_enabled(False)

def  load_example_input_datas():
    """fake data"""
    data_list = []
    input_1 = torch.randn(1, 3, 224, 224, dtype=torch.float32).cuda()
    data_list.append(input_1)
    return data_list


if __name__ == '__main__':

    input_datas = load_example_input_datas()
    original_model = models.resnet50(pretrained=True).cuda().eval()

    option = poros.PorosOptions()
    # option.max_workspace_size = 1 << 30
    # option.is_dynamic = False
    # option.debug = True
    # option.unconst_ops_thres = 0


    try:
        poros_model = poros.compile(torch.jit.script(original_model), input_datas, option)
    except Exception as e:
        print("compile poros_model failed. error msg: {}".format(e))
        exit(0)


    for input in input_datas:
        ori_res = original_model(input)
        poros_res = poros_model(input)
        res_diff = torch.abs(ori_res - poros_res)
        print("max_diff", torch.max(res_diff))
        print(poros_res.shape)

    # warm up
    for i in range (100):
        for input in input_datas:
            ori_res = original_model(input)
            poros_res = poros_model(input)

    count = 1000

    # POROS benchmark
    torch.cuda.synchronize()
    st = time.time()
    for i in range (count):
        # step4: 预测。
        for input in input_datas:
            poros_res = poros_model(input)

    torch.cuda.synchronize()
    poros_elapsed_time = time.time() - st
    print("poros infer time:{:.5f}ms/infer".format(poros_elapsed_time))

    # original benchmark
    torch.cuda.synchronize()
    st = time.time()
    for i in range (count):
        # step4: 预测。
        for input in input_datas:
            ori_res = original_model(input)

    torch.cuda.synchronize()
    original_elapsed_time = time.time() - st
    print("original infer time/:{:.5f}ms/infer".format(original_elapsed_time))
    print("speedup: +{:.2f}%".format((original_elapsed_time / poros_elapsed_time - 1 ) * 100))