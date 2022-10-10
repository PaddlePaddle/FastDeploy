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
void BindVPL(pybind11::module& m) {
  // Bind VPL
  pybind11::class_<vision::faceid::VPL,
                   vision::faceid::InsightFaceRecognitionModel>(m, "VPL")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::faceid::VPL& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::faceid::VPL::size)
      .def_readwrite("alpha", &vision::faceid::VPL::alpha)
      .def_readwrite("beta", &vision::faceid::VPL::beta)
      .def_readwrite("swap_rb", &vision::faceid::VPL::swap_rb)
      .def_readwrite("l2_normalize", &vision::faceid::VPL::l2_normalize);
}

}  // namespace fastdeploy
