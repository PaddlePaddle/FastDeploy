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
void BindCosFace(pybind11::module& m) {
  // Bind CosFace
  pybind11::class_<vision::faceid::CosFace,
                   vision::faceid::InsightFaceRecognitionModel>(m, "CosFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::faceid::CosFace& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::faceid::CosFace::size)
      .def_readwrite("alpha", &vision::faceid::CosFace::alpha)
      .def_readwrite("beta", &vision::faceid::CosFace::beta)
      .def_readwrite("swap_rb", &vision::faceid::CosFace::swap_rb)
      .def_readwrite("l2_normalize", &vision::faceid::CosFace::l2_normalize);
}

}  // namespace fastdeploy
