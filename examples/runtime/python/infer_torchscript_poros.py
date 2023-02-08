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

from fastdeploy import ModelFormat

import fastdeploy as fd
import numpy as np


def load_example_input_datas():
    """prewarm datas"""
    data_list = []
    # max size
    input_1 = np.ones((1, 3, 224, 224), dtype=np.float32)
    max_inputs = [input_1]
    data_list.append(tuple(max_inputs))

    # min size
    input_1 = np.ones((1, 3, 224, 224), dtype=np.float32)
    min_inputs = [input_1]
    data_list.append(tuple(min_inputs))

    # opt size
    input_1 = np.ones((1, 3, 224, 224), dtype=np.float32)
    opt_inputs = [input_1]
    data_list.append(tuple(opt_inputs))

    return data_list


if __name__ == '__main__':
    # prewarm_datas
    prewarm_datas = load_example_input_datas()
    # download model
    model_url = "https://bj.bcebos.com/paddlehub/fastdeploy/std_resnet50_script.pt"
    fd.download(model_url, path=".")

    option = fd.RuntimeOption()
    option.use_gpu(0)
    option.use_poros_backend()
    option.set_model_path(
        "std_resnet50_script.pt", model_format=ModelFormat.TORCHSCRIPT)
    # compile
    runtime = fd.Runtime(option)
    runtime.compile(prewarm_datas)

    # infer
    input_data_0 = np.random.rand(1, 3, 224, 224).astype("float32")
    result = runtime.forward(input_data_0)
    print(result[0].shape)
