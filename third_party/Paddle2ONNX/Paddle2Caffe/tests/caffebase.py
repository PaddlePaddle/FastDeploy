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

import os
import shutil
from inspect import isfunction
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)-5s%(message)s",
                    datefmt="20%y-%m-%d %H:%M:%S")
import paddle
os.environ['GLOG_minloglevel'] = '2'
import caffe

from paddle2caffe.convert import dygraph2caffe


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(result) == np.ndarray:
        if type(expect) == list:
            expect = expect[0]
        expect = np.array(expect)
        logging.info('\t caffe.shape:{} vs fluid.shape:{}'.format(result.shape, expect.shape))
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            if result.dtype == np.bool_:
                diff = abs(result.astype("int32") - expect.astype("int32"))
            else:
                diff = abs(result - expect)
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error("Different output data types! res type is: {}, "
                          "and expect type is: {}".format(result.dtype, expect.dtype))
        assert res
        assert result.shape == expect.shape, "result.shape: {} != expect.shape: {}".format(
            result.shape, expect.shape)
        assert result.dtype == expect.dtype, "result.dtype: {} != expect.dtype: {}".format(
            result.dtype, expect.dtype)
    elif type(result) == list and len(result) > 1:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                compare(result[i], expect[i], delta, rtol)
            else:
                compare(result[i].numpy(), expect[i], delta, rtol)
    elif len(result) == 1:
        compare(result[0], expect, delta, rtol)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)

    elif dtype == "bool":
        return np.random.randint(low, high, shape).astype("bool")


class BuildFunc(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, inner_func, **super_param):
        super(BuildFunc, self).__init__()
        self.inner_func = inner_func
        self._super_param = super_param

    def forward(self, inputs):
        """
        forward
        """
        x = self.inner_func(inputs, **self._super_param)
        return x


class BuildClass(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, inner_class, **super_param):
        super(BuildClass, self).__init__()
        self.inner_class = inner_class(**super_param)

    def forward(self, inputs):
        """
        forward
        """
        x = self.inner_class(inputs)
        return x


class APICaffe(object):
    """
     paddle API transfer to caffe
    """

    def __init__(self,
                 func,
                 file_name,
                 ops=[],
                 input_spec_shape=[],
                 delta=1e-5,
                 rtol=1e-5,
                 use_gpu=False,
                 **sup_params):
        self.ops = ops
        if isinstance(self.ops, str):
            self.ops = [self.ops]
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.func = func
        if use_gpu and paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu']
        else:
            self.places = ['cpu']
        self.name = file_name
        self.pwd = os.getcwd()
        self.delta = delta
        self.rtol = rtol
        self.static = False
        self.kwargs_dict = {"input_data": ()}
        self._shape = []
        self._dtype = []
        self.input_spec = []
        self.input_feed = {}
        self.input_spec_shape = input_spec_shape
        self.input_dtype = []

        if isfunction(self.func):
            # self._func = self.BuildFunc(self.func, **self.kwargs_dict_dygraph["params_group1"])
            self._func = BuildFunc(inner_func=self.func, **sup_params)
        elif isinstance(self.func, type):
            self._func = BuildClass(inner_class=self.func, **sup_params)
        else:
            self._func = self.func

    def set_input_data(self, group_name, *args):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = args
        if isinstance(self.kwargs_dict[group_name][0], tuple):
            self.kwargs_dict[group_name] = self.kwargs_dict[group_name][0]

        i = 0
        for in_data in self.kwargs_dict[group_name]:
            if isinstance(in_data, list):
                for tensor_data in in_data:
                    self.input_dtype.append(tensor_data.dtype)
                    self.input_spec.append(
                        paddle.static.InputSpec(
                            shape=tensor_data.shape,
                            dtype=tensor_data.dtype,
                            name='input' + str(i)))
                    self.input_feed['input' + str(i)] = tensor_data.numpy()
                    i += 1
            else:
                if isinstance(in_data, tuple):
                    in_data = in_data[0]
                self.input_dtype.append(in_data.dtype)
                self.input_spec.append(
                    paddle.static.InputSpec(
                        shape=in_data.shape, dtype=in_data.dtype, name='input' + str(i)))
                self.input_feed['input' + str(i)] = in_data.numpy()
                i += 1

    def set_device_mode(self, is_gpu=True):
        if paddle.device.is_compiled_with_cuda() is True and is_gpu:
            self.places = ['gpu']
        else:
            self.places = ['cpu']

    def set_input_spec(self):
        if len(self.input_spec_shape) == 0:
            return
        self.input_spec.clear()
        i = 0
        for shape in self.input_spec_shape:
            self.input_spec.append(
                paddle.static.InputSpec(
                    shape=shape, dtype=self.input_dtype[i], name='input' + str(i)))
            i += 1

    def _mkdir(self):
        """
        make dir to save all
        """
        save_path = os.path.join(self.pwd, self.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def _rmdir(self):
        """
        clear dir
        """
        save_path = os.path.join(self.pwd, self.name)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def _mk_dygraph_exp(self, instance):
        """
        make expect npy
        """
        return instance(*self.kwargs_dict["input_data"])

    def _dygraph_to_caffe(self, instance):
        """
        paddle dygraph layer to caffe
        """
        enable_dev_version = True
        if os.getenv("ENABLE_DEV", "OFF") == "OFF":
            enable_dev_version = False
        dygraph2caffe(instance, os.path.join(self.pwd, self.name, self.name),
                      input_spec=self.input_spec,
                      auto_update_opset=False,
                      enable_dev_version=enable_dev_version)

    def _dygraph_jit_save(self, instance):
        """
        paddle dygraph layer to paddle static
        """
        paddle.jit.save(
            instance,
            os.path.join(self.pwd, self.name, self.name + '_jit_save'),
            input_spec=self.input_spec)

    def _mk_caffe_res(self):
        """
        make caffe res
        """
        proto = os.path.join(self.pwd, self.name, self.name + '.prototxt')
        weight = os.path.join(self.pwd, self.name, self.name + '.caffemodel')
        caffe.set_mode_cpu()
        net = caffe.Net(proto, weight, caffe.TEST)
        for key, value in self.input_feed.items():
            for blob in net.blobs:
                if blob.startswith(key + '_idx'):
                    net.blobs[blob].data[...] = value
                    break
        caffe_output = net.forward()
        caffe_outs = []
        # for key, value in net.params.items():
        #     for idx in range(len(net.params[key])):
        #         print(key, idx)
        #         print(net.params[key][idx].data.shape)
        #         print(net.params[key][idx].data)
        for key, value in caffe_output.items():
            caffe_outs.append(value)
        return caffe_outs

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def dev_check_ops(self, op_name, model_file_path):
        from paddle.fluid.proto import framework_pb2
        prog = framework_pb2.ProgramDesc()

        with open(model_file_path, "rb") as f:
            prog.ParseFromString(f.read())

        ops = set()
        find = False
        for block in prog.blocks:
            for op in block.ops:
                op_type = op.type
                op_type = op_type.replace("depthwise_", "")
                if op_type == op_name:
                    find = True
        return find

    def run(self):
        """
        1. use dygraph layer to make exp
        2. dygraph layer to caffe
        3. use caffe to make res
        4. compare diff
        """
        self._mkdir()
        self.set_input_spec()
        for place in self.places:
            paddle.set_device(place)

            exp = self._mk_dygraph_exp(self._func)
            res_fict = None
            if os.getenv("ENABLE_DEV", "OFF") == "OFF":
                # export caffe models and make caffe res
                self._dygraph_to_caffe(instance=self._func)
                res_fict = self._mk_caffe_res()
                if exp.dtype != paddle.float32 and exp.dtype != paddle.float64:
                    logging.warning('dtype mismatch, caffe only have float type')
                    exp = exp.astype(paddle.float32)
                compare(res_fict, exp, delta=self.delta, rtol=self.rtol)

                # dygraph model jit save
                if self.static is True and place == 'gpu':
                    self._dygraph_jit_save(instance=self._func)
            elif os.getenv("ENABLE_DEV", "OFF") == "ON":
                raise NotImplementedError
            else:
                print("`export ENABLE_DEV=ON or export ENABLE_DEV=OFF`")
        self._rmdir()
