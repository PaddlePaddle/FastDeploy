#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
import unittest
import os
import time
import sys
import random
import math
import functools
import tempfile
import contextlib
import numpy as np
import paddle
from fake_quant import post_quant_fake

paddle.enable_static()

random.seed(0)
np.random.seed(0)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.model_name = None
        self.quantize_model_dir = "./quantized_models/"

    def run_program(self,
                    model_path,
                    model_filename='__model__',
                    params_filename='__params__',
                    threads_num=None):
        print("test model path:" + model_path)
        import onnxruntime as rt
        import paddle2onnx

        onnx_model = paddle2onnx.command.c_paddle_to_onnx(
            model_file=model_path + "/" + model_filename,
            params_file=model_path + "/" + params_filename,
            opset_version=13,
            enable_onnx_checker=True)
        sess_options = rt.SessionOptions()
        if threads_num is not None:
            sess_options.intra_op_num_threads = threads_num
        sess = rt.InferenceSession(
            onnx_model, sess_options, providers=['CPUExecutionProvider'])

        input_names = [input.name for input in sess.get_inputs()]
        input_shape = [input.shape for input in sess.get_inputs()]

        input_dict = {}
        for index in range(len(input_shape)):
            shape = input_shape[index]
            name = input_names[index]
            new_shape = []
            for i in shape:
                if isinstance(i, str) or i is None or i <= 0:
                    i = 1
                new_shape.append(i)
            input_dict[name] = np.ones(new_shape, dtype="float32")

        periods = []
        for i in range(1600):
            if i == 200:
                t1 = time.time()
            out = sess.run(None, input_dict)
        t2 = time.time()
        period = t2 - t1
        periods.append(period)

        latency = np.average(periods)
        return latency

    def generate_quantized_model(self,
                                 original_model_path,
                                 quantize_model_path,
                                 model_filename='__model__',
                                 params_filename='__params__'):
        paddle.enable_static()
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        post_quant_fake(
            executor=exe,
            model_dir=original_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            save_model_path=quantize_model_path,
            is_full_quantize=True,
            onnx_format=True)

    def run_test(self,
                 model_name,
                 model_filename='__model__',
                 params_filename='__params__',
                 threads_num=None):
        self.model_name = model_name
        origin_model_path = os.path.join(self.quantize_model_dir, model_name)

        print("Start FP32 inference for {0}  ...".format(model_name))
        fp32_latency = self.run_program(origin_model_path, model_filename,
                                        params_filename, threads_num)

        print("Start INT8 post training quantization for {0} ...".format(
            model_name))
        quantize_model_path = os.path.join(self.quantize_model_dir,
                                           model_name + "_quantized")
        self.generate_quantized_model(origin_model_path, quantize_model_path,
                                      model_filename, params_filename)

        print("Start INT8 inference for {0}  ...".format(model_name))
        if (".pdmodel" in model_filename):
            int8_latency = self.run_program(quantize_model_path, model_filename,
                                            params_filename, threads_num)
        else:
            int8_latency = self.run_program(quantize_model_path,
                                            "__model__.pdmodel",
                                            "__model__.pdiparams", threads_num)

        print("---Post training quantization---")
        print("FP32 lentency {0}: latency {1} s."
              .format(model_name, fp32_latency))
        print("INT8 {0}: latency {1} s.\n".format(model_name, int8_latency))
        sys.stdout.flush()

        latency_diff = int8_latency - fp32_latency
        self.assertLess(latency_diff, -0.1)


class TestPostTrainingE2eqONNXFormatFullQuant(TestPostTrainingQuantization):
    def test_post_training_e2eq_onnx_format_full_quant(self):
        model_name = "e2eq"
        self.run_test(model_name, threads_num=1)


class TestPostTrainingFeedasqONNXFormatFullQuant(TestPostTrainingQuantization):
    def test_post_training_feedasq_onnx_format_full_quant(self):
        model_name = "feedasq"
        self.run_test(model_name, threads_num=1)


class TestPostTrainingFeedasqNohadamaONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_feedasq_nohadama_onnx_format_full_quant(self):
        model_name = "feedasq_nohadama"
        self.run_test(model_name, threads_num=1)


class TestPostTrainingVideofeedasqONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_videofeedasq_onnx_format_full_quant(self):
        model_name = "videofeedasq"
        self.run_test(model_name, threads_num=1)


class TestPostTrainingLiminghaoONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_liminghao_onnx_format_full_quant(self):
        model_name = "liminghao"
        self.run_test(model_name, "model.pdmodel", "model.pdiparams", 1)


if __name__ == '__main__':
    unittest.main()
