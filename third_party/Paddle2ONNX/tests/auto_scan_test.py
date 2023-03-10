# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import os
import time
import logging
import paddle

import hypothesis
from hypothesis import given, settings, seed, reproduce_failure
import hypothesis.strategies as st
from onnxbase import APIOnnx, randtool
from itertools import product
import copy
from inspect import isfunction

paddle.set_device("cpu")

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
settings.register_profile(
    "dev",
    max_examples=1000,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
if float(os.getenv('TEST_NUM_PERCENT_CASES', default='1.0')) < 1 or \
    os.getenv('HYPOTHESIS_TEST_PROFILE', 'dev') == 'ci':
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


class BaseNet(paddle.nn.Layer):
    """
    define Net
    """

    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = copy.copy(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class OPConvertAutoScanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OPConvertAutoScanTest, self).__init__(*args, **kwargs)
        np.random.seed(1024)
        paddle.enable_static()
        self.num_ran_models = 0

    def run_and_statis(self,
                       max_examples=100,
                       opset_version=[7, 9, 15],
                       reproduce=None,
                       min_success_num=25,
                       max_duration=-1):
        if os.getenv("CE_STAGE", "OFF") == "ON":
            max_examples *= 10
            min_success_num *= 10
            # while at ce phase, there's no limit on time
            max_duration = -1
        start_time = time.time()
        settings.register_profile(
            "ci",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False, )
        settings.load_profile("ci")

        def sample_convert_generator(draw):
            return self.sample_convert_config(draw)

        def run_test(configs):
            return self.run_test(configs=configs)

        generator = st.composite(sample_convert_generator)
        loop_func = given(generator())(run_test)
        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info("Start to running test of {}".format(type(self)))

        paddle.disable_static()
        loop_func()

        logging.info(
            "===================Statistical Information===================")
        logging.info("Number of Generated Programs: {}".format(
            self.num_ran_models))
        successful_ran_programs = int(self.num_ran_models)
        if successful_ran_programs < min_success_num:
            logging.warning("satisfied_programs = ran_programs")
            logging.error(
                "At least {} programs need to ran successfully, but now only about {} programs satisfied.".
                format(min_success_num, successful_ran_programs))
            assert False
        used_time = time.time() - start_time
        logging.info("Used time: {} s".format(round(used_time, 2)))
        if max_duration > 0 and used_time > max_duration:
            logging.error(
                "The duration exceeds {} seconds, if this is neccessary, try to set a larger number for parameter `max_duration`.".
                format(max_duration))
            assert False

    def run_test(self, configs):
        config, models = configs
        logging.info("Run configs: {}".format(config))

        assert "op_names" in config.keys(
        ), "config must include op_names in dict keys"
        assert "test_data_shapes" in config.keys(
        ), "config must include test_data_shapes in dict keys"
        assert "test_data_types" in config.keys(
        ), "config must include test_data_types in dict keys"
        assert "opset_version" in config.keys(
        ), "config must include opset_version in dict keys"
        assert "input_spec_shape" in config.keys(
        ), "config must include input_spec_shape in dict keys"

        op_names = config["op_names"]
        test_data_shapes = config["test_data_shapes"]
        test_data_types = config["test_data_types"]
        opset_version = config["opset_version"]
        input_specs = config["input_spec_shape"]

        use_gpu = True
        if "use_gpu" in config.keys():
            use_gpu = config["use_gpu"]

        self.num_ran_models += 1

        if not isinstance(models, (tuple, list)):
            models = [models]
        if not isinstance(op_names, (tuple, list)):
            op_names = [op_names]
        if not isinstance(opset_version[0], (tuple, list)):
            opset_version = [opset_version]
        if len(opset_version) == 1 and len(models) != len(opset_version):
            opset_version = opset_version * len(models)

        assert len(models) == len(
            op_names), "Length of models should be equal to length of op_names"

        input_type_list = None
        if len(test_data_types) > 1:
            input_type_list = list(product(*test_data_types))
        elif len(test_data_types) == 1:
            if isinstance(test_data_types[0], str):
                input_type_list = [test_data_types[0]]
            else:
                input_type_list = test_data_types
        elif len(test_data_types) == 0:
            input_type_list = [["float32"] * len(test_data_shapes)]

        delta = 1e-5
        rtol = 1e-5
        if "delta" in config.keys():
            delta = config["delta"]
        if "rtol" in config.keys():
            rtol = config["rtol"]

        for i, model in enumerate(models):
            model.eval()
            obj = APIOnnx(model, op_names[i], opset_version[i], op_names[i],
                          input_specs, delta, rtol, use_gpu)
            for input_type in input_type_list:
                input_tensors = list()
                for j, shape in enumerate(test_data_shapes):
                    # Determine whether it is a user-defined data generation function
                    if isfunction(shape):
                        data = shape()
                        data = data.astype(input_type[j])
                        input_tensors.append(paddle.to_tensor(data))
                        continue
                    if input_type[j].count('int') > 0:
                        input_tensors.append(
                            paddle.to_tensor(
                                randtool("int", -20, 20, shape).astype(
                                    input_type[j])))
                    elif input_type[j].count('bool') > 0:
                        input_tensors.append(
                            paddle.to_tensor(
                                randtool("bool", -2, 2, shape).astype(
                                    input_type[j])))
                    else:
                        input_tensors.append(
                            paddle.to_tensor(
                                randtool("float", -2, 2, shape).astype(
                                    input_type[j])))
                obj.set_input_data("input_data", tuple(input_tensors))
                logging.info("Now Run >>> dtype: {}, op_name: {}".format(
                    input_type, op_names[i]))
                obj.run()
            if len(input_type_list) == 0:
                obj.run()
        logging.info("Run Successfully!")
