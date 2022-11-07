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

#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/pybind/main.h"

namespace fastdeploy {

void BindFDTensor(pybind11::module& m) {
  pybind11::class_<FDTensor>(m, "FDTensor")
      .def(pybind11::init<>(), "Default Constructor")
      .def_readwrite("name", &FDTensor::name)
      .def_readonly("shape", &FDTensor::shape)
      .def_readonly("dtype", &FDTensor::dtype)
      .def_readonly("device", &FDTensor::device)
      .def("numpy", [](FDTensor& self) {
        return TensorToPyArray(self);
      })
      .def("from_numpy", [](FDTensor& self, pybind11::array& pyarray, bool share_buffer = false) {
        PyArrayToTensor(pyarray, &self, share_buffer);
      });
}

}  // namespace fastdeploy
