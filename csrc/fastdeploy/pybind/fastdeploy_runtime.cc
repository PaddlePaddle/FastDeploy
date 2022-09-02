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
      .def("use_poros_backend", &RuntimeOption::UsePorosBackend)
      .def("use_ort_backend", &RuntimeOption::UseOrtBackend)
      .def("use_trt_backend", &RuntimeOption::UseTrtBackend)
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
      .def("set_input_dtypes", &RuntimeOption::SetInputDtypes)
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
                     &RuntimeOption::trt_max_workspace_size)
      .def_readwrite("is_dynamic", &RuntimeOption::is_dynamic)
      .def_readwrite("long_to_int", &RuntimeOption::long_to_int)
      .def_readwrite("use_nvidia_tf32", &RuntimeOption::use_nvidia_tf32)
      .def_readwrite("unconst_ops_thres", &RuntimeOption::unconst_ops_thres)
      .def_readwrite("poros_file", &RuntimeOption::poros_file);

  pybind11::class_<TensorInfo>(m, "TensorInfo")
      .def_readwrite("name", &TensorInfo::name)
      .def_readwrite("shape", &TensorInfo::shape)
      .def_readwrite("dtype", &TensorInfo::dtype);

  pybind11::class_<Runtime>(m, "Runtime")
      .def(pybind11::init())
      .def("init", &Runtime::Init)
      .def("compile", 
           [](Runtime& self, std::vector<std::vector<pybind11::array>>& warm_datas) {
             size_t rows = warm_datas.size();
             size_t columns = warm_datas[0].size();
             std::vector<std::vector<FDTensor>> warm_tensors(rows, std::vector<FDTensor>(columns));
             for (size_t i = 0; i < rows; ++i) {
                 for (size_t j = 0; j< columns; ++j) {
                     warm_tensors[i][j].dtype = NumpyDataTypeToFDDataType(warm_datas[i][j].dtype());
                     warm_tensors[i][j].shape.insert(
                        warm_tensors[i][j].shape.begin(), warm_datas[i][j].shape(),
                        warm_datas[i][j].shape() + warm_datas[i][j].ndim());
                     warm_tensors[i][j].data.resize(warm_datas[i][j].nbytes());
                     memcpy(warm_tensors[i][j].Data(), warm_datas[i][j].mutable_data(),
                        warm_datas[i][j].nbytes());
                 }
             }
             return self.Compile(warm_tensors);
           })
      .def("infer",
           [](Runtime& self, std::map<std::string, pybind11::array>& data) {
             std::vector<FDTensor> inputs(data.size());
             int index = 0;
             for (auto iter = data.begin(); iter != data.end(); ++iter) {
               inputs[index].dtype =
                   NumpyDataTypeToFDDataType(iter->second.dtype());
               inputs[index].shape.insert(
                   inputs[index].shape.begin(), iter->second.shape(),
                   iter->second.shape() + iter->second.ndim());
               // TODO(jiangjiajun) Maybe skip memory copy is a better choice
               // use SetExternalData
               inputs[index].data.resize(iter->second.nbytes());
               memcpy(inputs[index].Data(), iter->second.mutable_data(),
                      iter->second.nbytes());
               inputs[index].name = iter->first;
               index += 1;
             }

            //  std::vector<FDTensor> outputs(self.NumOutputs());
             std::vector<FDTensor> outputs(10);
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
      .value("POROS", Backend::POROS)
      .value("PDINFER", Backend::PDINFER);
  pybind11::enum_<Frontend>(m, "Frontend", pybind11::arithmetic(),
                            "Frontend for inference.")
      .value("PADDLE", Frontend::PADDLE)
      .value("TORCHSCRIPT", Frontend::TORCHSCRIPT)
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

  m.def("get_available_backends", []() { return GetAvailableBackends(); });
}

}  // namespace fastdeploy
