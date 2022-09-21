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
      .def("use_gpu", &RuntimeOption::UseGpu)
      .def("use_cpu", &RuntimeOption::UseCpu)
      .def("set_cpu_thread_num", &RuntimeOption::SetCpuThreadNum)
      .def("use_paddle_backend", &RuntimeOption::UsePaddleBackend)
      .def("use_ort_backend", &RuntimeOption::UseOrtBackend)
      .def("use_trt_backend", &RuntimeOption::UseTrtBackend)
      .def("use_openvino_backend", &RuntimeOption::UseOpenVINOBackend)
      .def("use_lite_backend", &RuntimeOption::UseLiteBackend)
      .def("enable_paddle_mkldnn", &RuntimeOption::EnablePaddleMKLDNN)
      .def("disable_paddle_mkldnn", &RuntimeOption::DisablePaddleMKLDNN)
      .def("enable_paddle_log_info", &RuntimeOption::EnablePaddleLogInfo)
      .def("disable_paddle_log_info", &RuntimeOption::DisablePaddleLogInfo)
      .def("set_paddle_mkldnn_cache_size",
           &RuntimeOption::SetPaddleMKLDNNCacheSize)
      .def("set_trt_input_shape", &RuntimeOption::SetTrtInputShape)
      .def("enable_trt_fp16", &RuntimeOption::EnableTrtFP16)
      .def("disable_trt_fp16", &RuntimeOption::DisableTrtFP16)
      .def("set_trt_cache_file", &RuntimeOption::SetTrtCacheFile)
      .def_readwrite("model_file", &RuntimeOption::model_file)
      .def_readwrite("params_file", &RuntimeOption::params_file)
      .def_readwrite("model_format", &RuntimeOption::model_format)
      .def_readwrite("backend", &RuntimeOption::backend)
      .def_readwrite("cpu_thread_num", &RuntimeOption::cpu_thread_num)
      .def_readwrite("device_id", &RuntimeOption::device_id)
      .def_readwrite("device", &RuntimeOption::device)
      .def_readwrite("ort_graph_opt_level", &RuntimeOption::ort_graph_opt_level)
      .def_readwrite("ort_inter_op_num_threads",
                     &RuntimeOption::ort_inter_op_num_threads)
      .def_readwrite("ort_execution_mode", &RuntimeOption::ort_execution_mode)
      .def_readwrite("trt_max_shape", &RuntimeOption::trt_max_shape)
      .def_readwrite("trt_opt_shape", &RuntimeOption::trt_opt_shape)
      .def_readwrite("trt_min_shape", &RuntimeOption::trt_min_shape)
      .def_readwrite("trt_serialize_file", &RuntimeOption::trt_serialize_file)
      .def_readwrite("trt_enable_fp16", &RuntimeOption::trt_enable_fp16)
      .def_readwrite("trt_enable_int8", &RuntimeOption::trt_enable_int8)
      .def_readwrite("trt_max_batch_size", &RuntimeOption::trt_max_batch_size)
      .def_readwrite("trt_max_workspace_size",
                     &RuntimeOption::trt_max_workspace_size);

  pybind11::class_<TensorInfo>(m, "TensorInfo")
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("dtype", &TensorInfo::dtype);

  pybind11::class_<Runtime>(m, "Runtime")
      .def(pybind11::init())
      .def("init", &Runtime::Init)
      .def("infer",
           [](Runtime& self, std::vector<FDTensor>& inputs) {
             std::vector<FDTensor> outputs(self.NumOutputs());
             self.Infer(inputs, &outputs);
             return outputs;
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
      .value("PDINFER", Backend::PDINFER)
      .value("LITE", Backend::LITE);
  pybind11::enum_<Frontend>(m, "Frontend", pybind11::arithmetic(),
                            "Frontend for inference.")
      .value("PADDLE", Frontend::PADDLE)
      .value("ONNX", Frontend::ONNX);
  pybind11::enum_<Device>(m, "Device", pybind11::arithmetic(),
                          "Device for inference.")
      .value("CPU", Device::CPU)
      .value("GPU", Device::GPU);

  pybind11::enum_<FDDataType>(m, "FDDataType", pybind11::arithmetic(),
                              "Data type of FastDeploy.")
      .value("BOOL", FDDataType::BOOL)
      .value("INT8", FDDataType::INT8)
      .value("INT16", FDDataType::INT16)
      .value("INT32", FDDataType::INT32)
      .value("INT64", FDDataType::INT64)
      .value("FP32", FDDataType::FP32)
      .value("FP64", FDDataType::FP64)
      .value("UINT8", FDDataType::UINT8);

  pybind11::class_<FDTensor>(m, "FDTensor", pybind11::buffer_protocol())
      .def(pybind11::init())
      .def("cpu_data",
           [](FDTensor& self) {
             auto ptr = self.CpuData();
             auto numel = self.Numel();
             auto dtype = FDDataTypeToNumpyDataType(self.dtype);
             auto base = pybind11::array(dtype, self.shape);
             return pybind11::array(dtype, self.shape, ptr, base);
           })
      .def("resize", static_cast<void (FDTensor::*)(size_t)>(&FDTensor::Resize))
      .def("resize",
           static_cast<void (FDTensor::*)(const std::vector<int64_t>&)>(
               &FDTensor::Resize))
      .def(
          "resize",
          [](FDTensor& self, const std::vector<int64_t>& shape,
             const FDDataType& dtype, const std::string& name,
             const Device& device) { self.Resize(shape, dtype, name, device); })
      .def("numel", &FDTensor::Numel)
      .def("nbytes", &FDTensor::Nbytes)
      .def_readwrite("name", &FDTensor::name)
      .def_readonly("shape", &FDTensor::shape)
      .def_readonly("dtype", &FDTensor::dtype)
      .def_readonly("device", &FDTensor::device);

  m.def("get_available_backends", []() { return GetAvailableBackends(); });
}

}  // namespace fastdeploy
