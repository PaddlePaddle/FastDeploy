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
#include "fastdeploy/fastdeploy_model.h"

namespace fastdeploy {

void BindFDModel(pybind11::module& m) {
  pybind11::class_<FastDeployModel>(m, "FastDeployModel")
      .def(pybind11::init<>(), "Default Constructor")
      .def("model_name", &FastDeployModel::ModelName)
      .def("num_inputs_of_runtime", &FastDeployModel::NumInputsOfRuntime)
      .def("num_outputs_of_runtime", &FastDeployModel::NumOutputsOfRuntime)
      .def("input_info_of_runtime", &FastDeployModel::InputInfoOfRuntime)
      .def("output_info_of_runtime", &FastDeployModel::OutputInfoOfRuntime)
      .def("initialized", &FastDeployModel::Initialized)
      .def_readwrite("runtime_option", &FastDeployModel::runtime_option)
      .def_readwrite("valid_cpu_backends", &FastDeployModel::valid_cpu_backends)
      .def_readwrite("valid_gpu_backends",
                     &FastDeployModel::valid_gpu_backends);
}

} // namespace fastdeploy
