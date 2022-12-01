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
from __future__ import absolute_import
import logging
import numpy as np
from . import ModelFormat
from . import c_lib_wrap as C
from . import rknpu2


class Runtime:
    """FastDeploy Runtime object.
    """

    def __init__(self, runtime_option):
        """Initialize a FastDeploy Runtime object.

        :param runtime_option: (fastdeploy.RuntimeOption)Options for FastDeploy Runtime
        """

        self._runtime = C.Runtime()
        self.runtime_option = runtime_option
        assert self._runtime.init(
            self.runtime_option._option), "Initialize Runtime Failed!"

    def forward(self, *inputs):
        """[Only for Poros backend] Inference with input data for poros

        :param data: (list[str : numpy.ndarray])The input data list
        :return list of numpy.ndarray
        """
        if self.runtime_option._option.model_format != ModelFormat.TORCHSCRIPT:
            raise Exception(
                "The forward function is only used for Poros backend, please call infer function"
            )
        inputs_dict = dict()
        for i in range(len(inputs)):
            inputs_dict["x" + str(i)] = inputs[i]
        return self.infer(inputs_dict)

    def infer(self, data):
        """Inference with input data.

        :param data: (dict[str : numpy.ndarray])The input data dict, key value must keep same with the loaded model
        :return list of numpy.ndarray
        """
        assert isinstance(data, dict) or isinstance(
            data, list), "The input data should be type of dict or list."
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, np.ndarray) and not v.data.contiguous:
                    data[k] = np.ascontiguousarray(data[k])

        return self._runtime.infer(data)

    def bind_input_tensor(self, name, fdtensor):
        """Bind FDTensor by name, no copy and share input memory

        :param name: (str)The name of input data.
        :param fdtensor: (fastdeploy.FDTensor)The input FDTensor.
        """
        self._runtime.bind_input_tensor(name, fdtensor)

    def zero_copy_infer(self):
        """No params inference the model.

        the input and output data need to pass through the bind_input_tensor and get_output_tensor interfaces.
        """
        self._runtime.infer()

    def get_output_tensor(self, name):
        """Get output FDTensor by name, no copy and share backend output memory

        :param name: (str)The name of output data.
        :return fastdeploy.FDTensor
        """
        return self._runtime.get_output_tensor(name)

    def compile(self, warm_datas):
        """[Only for Poros backend] compile with prewarm data for poros

        :param data: (list[str : numpy.ndarray])The prewarm data list
        :return TorchScript Model
        """
        if self.runtime_option._option.model_format != ModelFormat.TORCHSCRIPT:
            raise Exception(
                "The compile function is only used for Poros backend, please call infer function"
            )
        assert isinstance(warm_datas,
                          list), "The prewarm data should be type of list."
        for i in range(len(warm_datas)):
            warm_data = warm_datas[i]
            if isinstance(warm_data[0], np.ndarray):
                warm_data = list(data for data in warm_data)
            else:
                warm_data = list(data.numpy() for data in warm_data)
            warm_datas[i] = warm_data
        return self._runtime.compile(warm_datas, self.runtime_option._option)

    def num_inputs(self):
        """Get number of inputs of the loaded model.
        """
        return self._runtime.num_inputs()

    def num_outputs(self):
        """Get number of outputs of the loaded model.
        """
        return self._runtime.num_outputs()

    def get_input_info(self, index):
        """Get input information of the loaded model.

        :param index: (int)Index of the input
        :return fastdeploy.TensorInfo
        """
        assert isinstance(
            index, int), "The input parameter index should be type of int."
        assert index < self.num_inputs(
        ), "The input parameter index:{} should less than number of inputs:{}.".format(
            index, self.num_inputs)
        return self._runtime.get_input_info(index)

    def get_output_info(self, index):
        """Get output information of the loaded model.

        :param index: (int)Index of the output
        :return fastdeploy.TensorInfo
        """
        assert isinstance(
            index, int), "The input parameter index should be type of int."
        assert index < self.num_outputs(
        ), "The input parameter index:{} should less than number of outputs:{}.".format(
            index, self.num_outputs)
        return self._runtime.get_output_info(index)


class RuntimeOption:
    """Options for FastDeploy Runtime.
    """

    def __init__(self):
        """Initialize a FastDeploy RuntimeOption object.
        """

        self._option = C.RuntimeOption()

    @property
    def is_dynamic(self):
        """Only for Poros backend

        :param value: (bool)Whether to enable dynamic shape, default False
        """
        return self._option.is_dynamic

    @property
    def unconst_ops_thres(self):
        """Only for Poros backend

        :param value: (int)Minimum number of subgraph OPs, default 10
        """
        return self._option.unconst_ops_thres

    @property
    def long_to_int(self):
        """Only for Poros backend

        :param value: (bool)Whether to convert long dtype to int dtype, default True
        """
        return self._option.long_to_int

    @property
    def use_nvidia_tf32(self):
        """Only for Poros backend

        :param value: (bool)The calculation accuracy of tf32 mode exists on the A card, which can bring some performance improvements, default False
        """
        return self._option.use_nvidia_tf32

    @is_dynamic.setter
    def is_dynamic(self, value):
        assert isinstance(
            value, bool), "The value to set `is_dynamic` must be type of bool."
        self._option.is_dynamic = value

    @unconst_ops_thres.setter
    def unconst_ops_thres(self, value):
        assert isinstance(
            value,
            int), "The value to set `unconst_ops_thres` must be type of int."
        self._option.unconst_ops_thres = value

    @long_to_int.setter
    def long_to_int(self, value):
        assert isinstance(
            value,
            bool), "The value to set `long_to_int` must be type of bool."
        self._option.long_to_int = value

    @use_nvidia_tf32.setter
    def use_nvidia_tf32(self, value):
        assert isinstance(
            value,
            bool), "The value to set `use_nvidia_tf32` must be type of bool."
        self._option.use_nvidia_tf32 = value

    def set_model_path(self,
                       model_path,
                       params_path="",
                       model_format=ModelFormat.PADDLE):
        """Set path of model file and parameters file

        :param model_path: (str)Path of model file
        :param params_path: (str)Path of parameters file
        :param model_format: (ModelFormat)Format of model, support ModelFormat.PADDLE/ModelFormat.ONNX
        """
        return self._option.set_model_path(model_path, params_path,
                                           model_format)

    def use_gpu(self, device_id=0):
        """Inference with Nvidia GPU

        :param device_id: (int)The index of GPU will be used for inference, default 0
        """
        if not C.is_built_with_gpu():
            logging.warning(
                "The installed fastdeploy-python package is not built with GPU, will force to use CPU. To use GPU, following the commands to install fastdeploy-gpu-python."
            )
            logging.warning(
                "    ================= Install GPU FastDeploy===============")
            logging.warning("    python -m pip uninstall fastdeploy-python")
            logging.warning(
                "    python -m pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html"
            )
            return
        return self._option.use_gpu(device_id)

    def use_cpu(self):
        """Inference with CPU
        """
        return self._option.use_cpu()

    def use_rknpu2(self,
                   rknpu2_name=rknpu2.CpuName.RK3588,
                   rknpu2_core=rknpu2.CoreMask.RKNN_NPU_CORE_0):
        return self._option.use_rknpu2(rknpu2_name, rknpu2_core)

    def set_cpu_thread_num(self, thread_num=-1):
        """Set number of threads if inference with CPU

        :param thread_num: (int)Number of threads, if not positive, means the number of threads is decided by the backend, default -1
        """
        return self._option.set_cpu_thread_num(thread_num)

    def set_ort_graph_opt_level(self, level=-1):
        """Set graph optimization level for ONNX Runtime backend

        :param level: (int)Optimization level, -1 means the default setting
        """
        return self._option.set_ort_graph_opt_level(level)

    def use_paddle_backend(self):
        """Use Paddle Inference backend, support inference Paddle model on CPU/Nvidia GPU.
        """
        return self._option.use_paddle_backend()

    def use_paddle_infer_backend(self):
        """Wrapper function of use_paddle_backend(), use Paddle Inference backend, support inference Paddle model on CPU/Nvidia GPU.
        """
        return self.use_paddle_backend()

    def use_poros_backend(self):
        """Use Poros backend, support inference TorchScript model on CPU/Nvidia GPU.
        """
        return self._option.use_poros_backend()

    def use_ort_backend(self):
        """Use ONNX Runtime backend, support inference Paddle/ONNX model on CPU/Nvidia GPU.
        """
        return self._option.use_ort_backend()

    def use_trt_backend(self):
        """Use TensorRT backend, support inference Paddle/ONNX model on Nvidia GPU.
        """
        return self._option.use_trt_backend()

    def use_openvino_backend(self):
        """Use OpenVINO backend, support inference Paddle/ONNX model on CPU.
        """
        return self._option.use_openvino_backend()

    def use_lite_backend(self):
        """Use Paddle Lite backend, support inference Paddle model on ARM CPU.
        """
        return self._option.use_lite_backend()

    def use_paddle_lite_backend(self):
        """Wrapper function of use_lite_backend(), use Paddle Lite backend, support inference Paddle model on ARM CPU.
        """
        return self.use_lite_backend()

    def set_paddle_mkldnn(self, use_mkldnn=True):
        """Enable/Disable MKLDNN while using Paddle Inference backend, mkldnn is enabled by default.
        """
        return self._option.set_paddle_mkldnn(use_mkldnn)

    def set_openvino_device(self, name="CPU"):
        """Set device name for OpenVINO, default 'CPU', can also be 'AUTO', 'GPU', 'GPU.1'....
        """
        return self._option.set_openvino_device(name)

    def set_openvino_shape_info(self, shape_info):
        """Set shape information of the models' inputs, used for GPU to fix the shape

        :param shape_info: (dict{str, list of int})Shape information of model's inputs, e.g {"image": [1, 3, 640, 640], "scale_factor": [1, 2]}
        """
        return self._option.set_openvino_shape_info(shape_info)

    def set_openvino_cpu_operators(self, operators):
        """While using OpenVINO backend and intel GPU, this interface specifies unsupported operators to run on CPU

        :param operators: (list of string)list of operators' name, e.g ["MulticlasNms"]
        """
        return self._option.set_openvino_cpu_operators(operators)

    def enable_paddle_log_info(self):
        """Enable print out the debug log information while using Paddle Inference backend, the log information is disabled by default.
        """
        return self._option.enable_paddle_log_info()

    def disable_paddle_log_info(self):
        """Disable print out the debug log information while using Paddle Inference backend, the log information is disabled by default.
        """
        return self._option.disable_paddle_log_info()

    def set_paddle_mkldnn_cache_size(self, cache_size):
        """Set size of shape cache while using Paddle Inference backend with MKLDNN enabled, default will cache all the dynamic shape.
        """
        return self._option.set_paddle_mkldnn_cache_size(cache_size)

    def enable_lite_fp16(self):
        """Enable half precision inference while using Paddle Lite backend on ARM CPU, fp16 is disabled by default.
        """
        return self._option.enable_lite_fp16()

    def disable_lite_fp16(self):
        """Disable half precision inference while using Paddle Lite backend on ARM CPU, fp16 is disabled by default.
        """
        return self._option.disable_lite_fp16()

    def set_lite_power_mode(self, mode):
        """Set POWER mode while using Paddle Lite backend on ARM CPU.
        """
        return self._option.set_lite_power_mode(mode)

    def set_trt_input_shape(self,
                            tensor_name,
                            min_shape,
                            opt_shape=None,
                            max_shape=None):
        """Set shape range information while using TensorRT backend with loadding a model contains dynamic input shape. While inference with a new input shape out of the set shape range, the tensorrt engine will be rebuilt to expand the shape range information.

        :param tensor_name: (str)Name of input which has dynamic shape
        :param min_shape: (list of int)Minimum shape of the input, e.g [1, 3, 224, 224]
        :param opt_shape: (list of int)Optimize shape of the input, this offten set as the most common input shape, if set to None, it will keep same with min_shape
        :param max_shape: (list of int)Maximum shape of the input, e.g [8, 3, 224, 224], if set to None, it will keep same with the min_shape
        """
        if opt_shape is None and max_shape is None:
            opt_shape = min_shape
            max_shape = min_shape
        else:
            assert opt_shape is not None and max_shape is not None, "Set min_shape only, or set min_shape, opt_shape, max_shape both."
        return self._option.set_trt_input_shape(tensor_name, min_shape,
                                                opt_shape, max_shape)

    def set_trt_cache_file(self, cache_file_path):
        """Set a cache file path while using TensorRT backend. While loading a Paddle/ONNX model with set_trt_cache_file("./tensorrt_cache/model.trt"), if file `./tensorrt_cache/model.trt` exists, it will skip building tensorrt engine and load the cache file directly; if file `./tensorrt_cache/model.trt` doesn't exist, it will building tensorrt engine and save the engine as binary string to the cache file.

        :param cache_file_path: (str)Path of tensorrt cache file
        """
        return self._option.set_trt_cache_file(cache_file_path)

    def enable_trt_fp16(self):
        """Enable half precision inference while using TensorRT backend, notice that not all the Nvidia GPU support FP16, in those cases, will fallback to FP32 inference.
        """
        return self._option.enable_trt_fp16()

    def disable_trt_fp16(self):
        """Disable half precision inference while suing TensorRT backend.
        """
        return self._option.disable_trt_fp16()

    def enable_pinned_memory(self):
        """Enable pinned memory. Pinned memory can be utilized to speedup the data transfer between CPU and GPU. Currently it's only suppurted in TRT backend and Paddle Inference backend.
        """
        return self._option.enable_pinned_memory()

    def disable_pinned_memory(self):
        """Disable pinned memory.
        """
        return self._option.disable_pinned_memory()

    def enable_paddle_to_trt(self):
        """While using TensorRT backend, enable_paddle_to_trt() will change to use Paddle Inference backend, and use its integrated TensorRT instead.
        """
        return self._option.enable_paddle_to_trt()

    def set_trt_max_workspace_size(self, trt_max_workspace_size):
        """Set max workspace size while using TensorRT backend.
        """
        return self._option.set_trt_max_workspace_size(trt_max_workspace_size)

    def set_trt_max_batch_size(self, trt_max_batch_size):
        """Set max batch size while using TensorRT backend.
        """
        return self._option.set_trt_max_batch_size(trt_max_batch_size)

    def enable_paddle_trt_collect_shape(self):
        """Enable collect subgraph shape information while using Paddle Inference with TensorRT
        """
        return self._option.enable_paddle_trt_collect_shape()

    def disable_paddle_trt_collect_shape(self):
        """Disable collect subgraph shape information while using Paddle Inference with TensorRT
        """
        return self._option.disable_paddle_trt_collect_shape()

    def use_ipu(self,
                device_num=1,
                micro_batch_size=1,
                enable_pipelining=False,
                batches_per_step=1):
        return self._option.use_ipu(device_num, micro_batch_size,
                                    enable_pipelining, batches_per_step)

    def set_ipu_config(self,
                       enable_fp16=False,
                       replica_num=1,
                       available_memory_proportion=1.0,
                       enable_half_partial=False):
        return self._option.set_ipu_config(enable_fp16, replica_num,
                                           available_memory_proportion,
                                           enable_half_partial)

    def __repr__(self):
        attrs = dir(self._option)
        message = "RuntimeOption(\n"
        for attr in attrs:
            if attr.startswith("__"):
                continue
            if hasattr(getattr(self._option, attr), "__call__"):
                continue
            message += "  {} : {}\t\n".format(attr,
                                              getattr(self._option, attr))
        message.strip("\n")
        message += ")"
        return message
