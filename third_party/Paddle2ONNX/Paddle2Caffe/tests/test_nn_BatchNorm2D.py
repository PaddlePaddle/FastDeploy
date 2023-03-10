# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
from caffebase import APICaffe
from caffebase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.BatchNorm2D(num_features=5, use_global_stats=True)

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        return x


def test_BatchNorm2D():
    """
    api: paddle.nn.BatchNorm2D
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_BatchNorm2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [1, 5, 10, 10]).astype('float32')))
    obj.run()


if __name__ == "__main__":
    test_BatchNorm2D()
