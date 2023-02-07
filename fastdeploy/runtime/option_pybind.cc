// Cropyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fastdeploy/pybind/main.h"

namespace fastdeploy {

void BindLiteOption(pybind11::module& m);
void BindOpenVINOOption(pybind11::module& m);
void BindOrtOption(pybind11::module& m);
void BindPorosOption(pybind11::module& m);

void BindOption(pybind11::module& m) {
  BindLiteOption(m);
  BindOpenVINOOption(m);
  BindOrtOption(m);
  BindPorosOption(m);

  pybind11::class_<RuntimeOption>(m, "RuntimeOption")
      .def(pybind11::init())
      .def("set_model_path", &RuntimeOption::SetModelPath)
      .def("set_model_buffer", &RuntimeOption::SetModelBuffer)
      .def("use_gpu", &RuntimeOption::UseGpu)
      .def("use_cpu", &RuntimeOption::UseCpu)
      .def("use_rknpu2", &RuntimeOption::UseRKNPU2)
      .def("use_sophgo", &RuntimeOption::UseSophgo)
      .def("use_ascend", &RuntimeOption::UseAscend)
      .def("use_kunlunxin", &RuntimeOption::UseKunlunXin)
      .def_readwrite("paddle_lite_option", &RuntimeOption::paddle_lite_option)
      .def_readwrite("openvino_option", &RuntimeOption::openvino_option)
      .def_readwrite("ort_option", &RuntimeOption::ort_option)
      .def_readwrite("poros_option", &RuntimeOption::poros_option)
      .def("set_external_stream", &RuntimeOption::SetExternalStream)
      .def("set_cpu_thread_num", &RuntimeOption::SetCpuThreadNum)
      .def("use_paddle_backend", &RuntimeOption::UsePaddleBackend)
      .def("use_poros_backend", &RuntimeOption::UsePorosBackend)
      .def("use_ort_backend", &RuntimeOption::UseOrtBackend)
      .def("set_ort_graph_opt_level", &RuntimeOption::SetOrtGraphOptLevel)
      .def("use_trt_backend", &RuntimeOption::UseTrtBackend)
      .def("use_openvino_backend", &RuntimeOption::UseOpenVINOBackend)
      .def("use_lite_backend", &RuntimeOption::UseLiteBackend)
      .def("set_lite_device_names", &RuntimeOption::SetLiteDeviceNames)
      .def("set_lite_context_properties",
           &RuntimeOption::SetLiteContextProperties)
      .def("set_lite_model_cache_dir", &RuntimeOption::SetLiteModelCacheDir)
      .def("set_lite_dynamic_shape_info",
           &RuntimeOption::SetLiteDynamicShapeInfo)
      .def("set_lite_subgraph_partition_path",
           &RuntimeOption::SetLiteSubgraphPartitionPath)
      .def("set_lite_mixed_precision_quantization_config_path",
           &RuntimeOption::SetLiteMixedPrecisionQuantizationConfigPath)
      .def("set_lite_subgraph_partition_config_buffer",
           &RuntimeOption::SetLiteSubgraphPartitionConfigBuffer)
      .def("set_paddle_mkldnn", &RuntimeOption::SetPaddleMKLDNN)
      .def("set_openvino_device", &RuntimeOption::SetOpenVINODevice)
      .def("set_openvino_shape_info", &RuntimeOption::SetOpenVINOShapeInfo)
      .def("set_openvino_cpu_operators",
           &RuntimeOption::SetOpenVINOCpuOperators)
      .def("enable_paddle_log_info", &RuntimeOption::EnablePaddleLogInfo)
      .def("disable_paddle_log_info", &RuntimeOption::DisablePaddleLogInfo)
      .def("set_paddle_mkldnn_cache_size",
           &RuntimeOption::SetPaddleMKLDNNCacheSize)
      .def("enable_lite_fp16", &RuntimeOption::EnableLiteFP16)
      .def("disable_lite_fp16", &RuntimeOption::DisableLiteFP16)
      .def("set_lite_power_mode", &RuntimeOption::SetLitePowerMode)
      .def("set_trt_input_shape", &RuntimeOption::SetTrtInputShape)
      .def("set_trt_max_workspace_size", &RuntimeOption::SetTrtMaxWorkspaceSize)
      .def("set_trt_max_batch_size", &RuntimeOption::SetTrtMaxBatchSize)
      .def("enable_paddle_to_trt", &RuntimeOption::EnablePaddleToTrt)
      .def("enable_trt_fp16", &RuntimeOption::EnableTrtFP16)
      .def("disable_trt_fp16", &RuntimeOption::DisableTrtFP16)
      .def("set_trt_cache_file", &RuntimeOption::SetTrtCacheFile)
      .def("enable_pinned_memory", &RuntimeOption::EnablePinnedMemory)
      .def("disable_pinned_memory", &RuntimeOption::DisablePinnedMemory)
      .def("enable_paddle_trt_collect_shape",
           &RuntimeOption::EnablePaddleTrtCollectShape)
      .def("disable_paddle_trt_collect_shape",
           &RuntimeOption::DisablePaddleTrtCollectShape)
      .def("use_ipu", &RuntimeOption::UseIpu)
      .def("set_ipu_config", &RuntimeOption::SetIpuConfig)
      .def("delete_paddle_backend_pass",
           &RuntimeOption::DeletePaddleBackendPass)
      .def("enable_profiling", &RuntimeOption::EnableProfiling)
      .def("disable_profiling", &RuntimeOption::DisableProfiling)
      .def("disable_paddle_trt_ops", &RuntimeOption::DisablePaddleTrtOPs)
      .def_readwrite("model_file", &RuntimeOption::model_file)
      .def_readwrite("params_file", &RuntimeOption::params_file)
      .def_readwrite("model_format", &RuntimeOption::model_format)
      .def_readwrite("backend", &RuntimeOption::backend)
      .def_readwrite("external_stream", &RuntimeOption::external_stream_)
      .def_readwrite("model_from_memory", &RuntimeOption::model_from_memory_)
      .def_readwrite("cpu_thread_num", &RuntimeOption::cpu_thread_num)
      .def_readwrite("device_id", &RuntimeOption::device_id)
      .def_readwrite("device", &RuntimeOption::device)
      .def_readwrite("trt_max_shape", &RuntimeOption::trt_max_shape)
      .def_readwrite("trt_opt_shape", &RuntimeOption::trt_opt_shape)
      .def_readwrite("trt_min_shape", &RuntimeOption::trt_min_shape)
      .def_readwrite("trt_serialize_file", &RuntimeOption::trt_serialize_file)
      .def_readwrite("trt_enable_fp16", &RuntimeOption::trt_enable_fp16)
      .def_readwrite("trt_enable_int8", &RuntimeOption::trt_enable_int8)
      .def_readwrite("trt_max_batch_size", &RuntimeOption::trt_max_batch_size)
      .def_readwrite("trt_max_workspace_size",
                     &RuntimeOption::trt_max_workspace_size)
      .def_readwrite("ipu_device_num", &RuntimeOption::ipu_device_num)
      .def_readwrite("ipu_micro_batch_size",
                     &RuntimeOption::ipu_micro_batch_size)
      .def_readwrite("ipu_enable_pipelining",
                     &RuntimeOption::ipu_enable_pipelining)
      .def_readwrite("ipu_batches_per_step",
                     &RuntimeOption::ipu_batches_per_step)
      .def_readwrite("ipu_enable_fp16", &RuntimeOption::ipu_enable_fp16)
      .def_readwrite("ipu_replica_num", &RuntimeOption::ipu_replica_num)
      .def_readwrite("ipu_available_memory_proportion",
                     &RuntimeOption::ipu_available_memory_proportion)
      .def_readwrite("ipu_enable_half_partial",
                     &RuntimeOption::ipu_enable_half_partial);
}
}  // namespace fastdeploy
