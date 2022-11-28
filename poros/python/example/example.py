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

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import poros
import numpy as np

def load_example_model():
    """load model示例，正常load model即可"""
    import torchvision.models as models
    std_resnet = models.resnet50(pretrained=True)
    std_resnet.cuda()
    std_resnet.eval()
    return std_resnet


def  load_example_input_datas():
    """加载预热数据"""
    data_list = []
    #max size
    input_1 = np.ones((3, 3, 96, 320), np.float32)
    input_tensor = torch.from_numpy(input_1).cuda()
    data_list.append(input_tensor)

    #min size
    input_2 = np.ones((1, 3, 96, 320), np.float32)
    input_tensor2 = torch.from_numpy(input_2).cuda()
    data_list.append(input_tensor2)
    
    #opt size
    input_3 = np.ones((1, 3, 96, 320), np.float32)
    input_tensor3 = torch.from_numpy(input_3).cuda()
    data_list.append(input_tensor3)

    return data_list


if __name__ == '__main__':
    print("this is an example for poros")

    # step1: 按照正常的torch模块的步骤，load模型和参数，此处以resnet50为例
    # load_example_model 过程中load的原始pytorch模型（python代码），必须是完成了poros预处理的python代码
    # poros预处理相关wiki: 【待补充】
    original_model = load_example_model()

    # step2: 准备预热数据。
    # 请准备 1-3 份内容不一样的预热数据(example中是准备了3份一样的预热数据，只是示例，实际中尽量不要这样做)
    # 每一份预热数据用tuple封装，除非该模型只有一个输入，且这个输入类型是torch.Tensor.
    # 多份预热数据用list连接。
    # ！！！注意: 预热数据是必须的。
    input_datas = load_example_input_datas()

    # step3: 调用poros，编译原始的model，得到PorosModel
    # 当 option.is_dynamic 为true时，设置的预热数据的个数必须为3的倍数。
    # 当 option.is_dynamic 为false是，设置的预热数据至少为1份。
    option = poros.PorosOptions()
    option.is_dynamic = True
    #option.debug = True

    try:
        poros_model = poros.compile(original_model, input_datas, option)
    except Exception as e:
        print("compile poros_model failed. error msg: {}".format(e))
        #poros_model = original_model
        exit(0)
    
    # 序列化&反序列化 
    # poros.save(poros_model, "poros_model.pt")
    # poros_model = poros.load("poros_model.pt", option)

    # 准备测试用的batch数据
    input = np.ones((3, 3, 96, 320), np.float32)
    batch_tensor = torch.from_numpy(input).cuda()

    # step4: 预测。
    #result = poros_model(input_datas[0])
    result = poros_model(batch_tensor)
    #result = original_model(batch_tensor)

    print(result.size())
    print(result)
