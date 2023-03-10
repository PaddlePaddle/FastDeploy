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
import paddle.fluid as fluid
from paddle.dataset.common import download
from paddle.static.quantization import PostTrainingQuantization

paddle.enable_static()

random.seed(0)
np.random.seed(0)


class TestPostTrainingQuantization(unittest.TestCase):
    def setUp(self):
        self.int8_model_path = "./post_training_quantization"
        self.download_path = 'int8/download'
        self.cache_folder = os.path.expanduser('~/.cache/paddle/dataset/' +
                                               self.download_path)
        try:
            os.system("mkdir -p " + self.int8_model_path)
        except Exception as e:
            print("Failed to create {} due to {}".format(self.int8_model_path,
                                                         str(e)))
            sys.exit(-1)

    def tearDown(self):
        pass

    def merge_params(self, input_model_path, output_model_path):
        import paddle.fluid as fluid
        import paddle
        import sys
        paddle.enable_static()
        model_dir = input_model_path
        new_model_dir = output_model_path
        exe = fluid.Executor(fluid.CPUPlace())
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname=model_dir, executor=exe)

        fluid.io.save_inference_model(
            dirname=new_model_dir,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=inference_program,
            params_filename="__params__")

    def run_program(self,
                    model_path,
                    batch_size,
                    infer_iterations,
                    model_filename="",
                    params_filename="",
                    use_onnxruntime=False):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        infer_program = None
        feed_dict = None
        fetch_targets = None
        input_name = None
        sess = None
        if use_onnxruntime:
            import onnxruntime as rt
            import paddle2onnx
            new_model_path = model_path
            if model_filename == "":
                new_model_path = model_path + "_conbined"
                self.merge_params(model_path, new_model_path)
                model_filename = "__model__"
                params_filename = "__params__"
            onnx_model = paddle2onnx.command.c_paddle_to_onnx(
                model_file=new_model_path + "/" + model_filename,
                params_file=new_model_path + "/" + params_filename,
                opset_version=13,
                enable_onnx_checker=True)
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess = rt.InferenceSession(
                onnx_model, sess_options, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
        else:
            new_model_path = model_path
            if model_filename == "":
                new_model_path = model_path + "_conbined"
                self.merge_params(model_path, new_model_path)
                model_filename = "__model__"
                params_filename = "__params__"
            [infer_program, feed_dict,
             fetch_targets] = fluid.io.load_inference_model(
                 new_model_path,
                 exe,
                 model_filename=model_filename,
                 params_filename=params_filename)

        val_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size)

        img_shape = [1, 28, 28]
        test_info = []
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            image = np.array([x[0].reshape(img_shape)
                              for x in data]).astype("float32")
            input_label = np.array([x[1] for x in data]).astype("int64")

            t1 = time.time()
            if use_onnxruntime:
                out = sess.run(None, {input_name: image})
            else:
                out = exe.run(infer_program,
                              feed={feed_dict[0]: image},
                              fetch_list=fetch_targets)
            t2 = time.time()
            period = t2 - t1
            periods.append(period)

            out_label = np.argmax(np.array(out[0]), axis=1)
            top1_num = sum(input_label == out_label)
            test_info.append(top1_num)
            cnt += len(data)

            if (batch_id + 1) == infer_iterations:
                break

        throughput = cnt / np.sum(periods)
        latency = np.average(periods)
        acc1 = np.sum(test_info) / cnt
        return (throughput, latency, acc1)

    def generate_quantized_model(self,
                                 model_path,
                                 algo="KL",
                                 quantizable_op_type=["conv2d"],
                                 is_full_quantize=False,
                                 is_use_cache_file=False,
                                 is_optimize_model=False,
                                 batch_size=10,
                                 batch_nums=10,
                                 onnx_format=False,
                                 skip_tensor_list=None):

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        val_reader = paddle.dataset.mnist.train()
        new_model_path = model_path + "_conbined"
        self.merge_params(model_path, new_model_path)
        ptq = PostTrainingQuantization(
            executor=exe,
            model_dir=new_model_path,
            model_filename=new_model_path + "/__model__",
            params_filename=new_model_path + "/__params__",
            sample_generator=val_reader,
            batch_size=batch_size,
            batch_nums=batch_nums,
            algo=algo,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            optimize_model=is_optimize_model,
            onnx_format=onnx_format,
            skip_tensor_list=skip_tensor_list,
            is_use_cache_file=is_use_cache_file)
        ptq.quantize()
        ptq.save_quantized_model(
            self.int8_model_path,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams')

    def run_test(self,
                 model_name,
                 algo,
                 quantizable_op_type,
                 is_full_quantize,
                 is_use_cache_file,
                 is_optimize_model,
                 diff_threshold,
                 batch_size=10,
                 infer_iterations=10,
                 quant_iterations=5,
                 onnx_format=False,
                 skip_tensor_list=None):
        origin_model_path = os.path.join(self.cache_folder, model_name)

        print("Start FP32 inference for {0} on {1} images ...".format(
            model_name, infer_iterations * batch_size))
        (fp32_throughput, fp32_latency, fp32_acc1) = self.run_program(
            origin_model_path, batch_size, infer_iterations)

        print("Start FP32 inference on onnxruntime for {0} on {1} images ...".
              format(model_name, infer_iterations * batch_size))
        (onnx_fp32_throughput, onnx_fp32_latency,
         onnx_fp32_acc1) = self.run_program(
             origin_model_path,
             batch_size,
             infer_iterations,
             use_onnxruntime=True)

        print("Start INT8 post training quantization for {0} on {1} images ...".
              format(model_name, quant_iterations * batch_size))
        self.generate_quantized_model(
            origin_model_path, algo, quantizable_op_type, is_full_quantize,
            is_use_cache_file, is_optimize_model, batch_size, quant_iterations,
            onnx_format, skip_tensor_list)

        print("Start INT8 inference for {0} on {1} images ...".format(
            model_name, infer_iterations * batch_size))
        (int8_throughput, int8_latency, int8_acc1) = self.run_program(
            self.int8_model_path,
            batch_size,
            infer_iterations,
            model_filename="model.pdmodel",
            params_filename="model.pdiparams", )

        print("Start INT8 inference on onnxruntime for {0} on {1} images ...".
              format(model_name, infer_iterations * batch_size))
        (onnx_int8_throughput, onnx_int8_latency,
         onnx_int8_acc1) = self.run_program(
             self.int8_model_path,
             batch_size,
             infer_iterations,
             model_filename="model.pdmodel",
             params_filename="model.pdiparams",
             use_onnxruntime=True)

        print("---Post training quantization of {} method---".format(algo))
        print(
            "FP32 {0}: batch_size {1}, throughput {2} img/s, latency {3} s, acc1 {4}."
            .format(model_name, batch_size, fp32_throughput, fp32_latency,
                    fp32_acc1))
        print(
            "ONNXRuntime FP32 {0}: batch_size {1}, throughput {2} img/s, latency {3} s, acc1 {4}."
            .format(model_name, batch_size, onnx_fp32_throughput,
                    onnx_fp32_latency, onnx_fp32_acc1))
        print(
            "INT8 {0}: batch_size {1}, throughput {2} img/s, latency {3} s, acc1 {4}.\n"
            .format(model_name, batch_size, int8_throughput, int8_latency,
                    int8_acc1))
        print(
            "ONNXRuntime INT8 {0}: batch_size {1}, throughput {2} img/s, latency {3} s, acc1 {4}.\n"
            .format(model_name, batch_size, onnx_int8_throughput,
                    onnx_int8_latency, onnx_int8_acc1))
        sys.stdout.flush()

        # fp32 and int8
        delta_value = fp32_acc1 - int8_acc1
        self.assertLess(delta_value, diff_threshold)
        # onnx fp32 and paddle fp32
        delta_value = onnx_fp32_acc1 - fp32_acc1
        self.assertLess(delta_value, diff_threshold)
        # onnx int8 and paddle int8
        delta_value = onnx_fp32_acc1 - fp32_acc1
        self.assertLess(delta_value, diff_threshold)


class TestPostTrainingMseForMnistONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_mse_onnx_format_full_quant(self):
        model_name = "mnist_model"
        algo = "mse"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = True
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format)


class TestPostTrainingHistForMnistONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_hist_onnx_format_full_quant(self):
        model_name = "mnist_model"
        algo = "hist"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = True
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format)


class TestPostTrainingAvgForMnistONNXFormatFullQuant(
        TestPostTrainingQuantization):
    def test_post_training_abg_onnx_format_full_quant(self):
        model_name = "mnist_model"
        algo = "avg"
        quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
        is_full_quantize = True
        is_use_cache_file = False
        is_optimize_model = False
        onnx_format = True
        diff_threshold = 0.01
        batch_size = 10
        infer_iterations = 50
        quant_iterations = 5
        self.run_test(
            model_name,
            algo,
            quantizable_op_type,
            is_full_quantize,
            is_use_cache_file,
            is_optimize_model,
            diff_threshold,
            batch_size,
            infer_iterations,
            quant_iterations,
            onnx_format=onnx_format)


if __name__ == '__main__':
    unittest.main()
