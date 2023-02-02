// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

void BindRuntime(pybind11::module& m) {
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
      .def_readwrite("is_dynamic", &RuntimeOption::is_dynamic)
      .def_readwrite("long_to_int", &RuntimeOption::long_to_int)
      .def_readwrite("use_nvidia_tf32", &RuntimeOption::use_nvidia_tf32)
      .def_readwrite("unconst_ops_thres", &RuntimeOption::unconst_ops_thres)
      .def_readwrite("poros_file", &RuntimeOption::poros_file)
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

  pybind11::class_<TensorInfo>(m, "TensorInfo")
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("dtype", &TensorInfo::dtype);

  pybind11::class_<Runtime>(m, "Runtime")
      .def(pybind11::init())
      .def("init", &Runtime::Init)
      .def("compile",
           [](Runtime& self,
              std::vector<std::vector<pybind11::array>>& warm_datas,
              const RuntimeOption& _option) {
             size_t rows = warm_datas.size();
             size_t columns = warm_datas[0].size();
             std::vector<std::vector<FDTensor>> warm_tensors(
                 rows, std::vector<FDTensor>(columns));
             for (size_t i = 0; i < rows; ++i) {
               for (size_t j = 0; j < columns; ++j) {
                 auto dtype =
                     NumpyDataTypeToFDDataType(warm_datas[i][j].dtype());
                 std::vector<int64_t> data_shape;
                 data_shape.insert(
                     data_shape.begin(), warm_datas[i][j].shape(),
                     warm_datas[i][j].shape() + warm_datas[i][j].ndim());
                 warm_tensors[i][j].Resize(data_shape, dtype);
                 memcpy(warm_tensors[i][j].MutableData(),
                        warm_datas[i][j].mutable_data(),
                        warm_datas[i][j].nbytes());
               }
             }
             return self.Compile(warm_tensors, _option);
           })
      .def("infer",
           [](Runtime& self, std::map<std::string, pybind11::array>& data) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               std::vector<int64_t> data_shape;
               data_shape.insert(data_shape.begin(), iter->second.shape(),
                                 iter->second.shape() + iter->second.ndim());
               auto dtype = NumpyDataTypeToFDDataType(iter->second.dtype());
               // TODO(jiangjiajun) Maybe skip memory copy is a better choice
               // use SetExternalData
               inputs[index].Resize(data_shape, dtype);
               memcpy(inputs[index].MutableData(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }

             std::vector<FDTensor> outputs(self.NumOutputs());
             self.Infer(inputs, &outputs);

             std::vector<pybind11::array> results;
             results.reserve(outputs.size());
             for (size_t i = 0; i < outputs.size(); ++i) {
               auto numpy_dtype = FDDataTypeToNumpyDataType(outputs[i].dtype);
               results.emplace_back(
                   pybind11::array(numpy_dtype, outputs[i].shape));
               memcpy(results[i].mutable_data(), outputs[i].Data(),
                      outputs[i].Numel() * FDDataTypeSize(outputs[i].dtype));
             }
             return results;
           })
      .def("infer",
           [](Runtime& self, std::map<std::string, FDTensor>& data) {
             std::vector<FDTensor> inputs;
             inputs.reserve(data.size());
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               FDTensor tensor;
               tensor.SetExternalData(iter->second.Shape(),
                                      iter->second.Dtype(), iter->second.Data(),
                                      iter->second.device);
               tensor.name = iter->first;
               inputs.push_back(tensor);
             }
             std::vector<FDTensor> outputs;
             if (!self.Infer(inputs, &outputs)) {
               throw std::runtime_error("Failed to inference with Runtime.");
             }
             return outputs;
           })
      .def("infer",
           [](Runtime& self, std::vector<FDTensor>& inputs) {
             std::vector<FDTensor> outputs;
             self.Infer(inputs, &outputs);
             return outputs;
           })
      .def("bind_input_tensor", &Runtime::BindInputTensor)
      .def("infer", [](Runtime& self) { self.Infer(); })
      .def("get_output_tensor",
           [](Runtime& self, const std::string& name) {
             FDTensor* output = self.GetOutputTensor(name);
             if (output == nullptr) {
               return pybind11::cast(nullptr);
             }
             return pybind11::cast(*output);
           })
      .def("num_inputs", &Runtime::NumInputs)
      .def("num_outputs", &Runtime::NumOutputs)
      .def("get_input_info", &Runtime::GetInputInfo)
      .def("get_output_info", &Runtime::GetOutputInfo)
      .def_readonly("option", &Runtime::option);

  pybind11::enum_<Backend>(m, "Backend", pybind11::arithmetic(),
                           "Backend for inference.")
      .value("UNKOWN", Backend::UNKNOWN)
      .value("ORT", Backend::ORT)
      .value("TRT", Backend::TRT)
      .value("POROS", Backend::POROS)
      .value("PDINFER", Backend::PDINFER)
      .value("RKNPU2", Backend::RKNPU2)
      .value("SOPHGOTPU", Backend::SOPHGOTPU)
      .value("LITE", Backend::LITE);
  pybind11::enum_<ModelFormat>(m, "ModelFormat", pybind11::arithmetic(),
                               "ModelFormat for inference.")
      .value("PADDLE", ModelFormat::PADDLE)
      .value("TORCHSCRIPT", ModelFormat::TORCHSCRIPT)
      .value("RKNN", ModelFormat::RKNN)
      .value("SOPHGO", ModelFormat::SOPHGO)
      .value("ONNX", ModelFormat::ONNX);
  pybind11::enum_<Device>(m, "Device", pybind11::arithmetic(),
                          "Device for inference.")
      .value("CPU", Device::CPU)
      .value("GPU", Device::GPU)
      .value("IPU", Device::IPU)
      .value("RKNPU", Device::RKNPU)
      .value("SOPHGOTPU", Device::SOPHGOTPUD);

  pybind11::enum_<FDDataType>(m, "FDDataType", pybind11::arithmetic(),
                              "Data type of FastDeploy.")
      .value("BOOL", FDDataType::BOOL)
      .value("INT8", FDDataType::INT8)
      .value("INT16", FDDataType::INT16)
      .value("INT32", FDDataType::INT32)
      .value("INT64", FDDataType::INT64)
      .value("FP16", FDDataType::FP16)
      .value("FP32", FDDataType::FP32)
      .value("FP64", FDDataType::FP64)
      .value("UINT8", FDDataType::UINT8);

  m.def("get_available_backends", []() { return GetAvailableBackends(); });
}

}  // namespace fastdeploy
