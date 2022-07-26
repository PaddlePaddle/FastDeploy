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
void BindPPSeg(pybind11::module& m) {
  auto ppseg_module =
      m.def_submodule("ppseg", "Module to deploy PaddleSegmentation.");
  pybind11::class_<vision::ppseg::Model, FastDeployModel>(ppseg_module, "Model")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict", [](vision::ppseg::Model& self, pybind11::array& data) {
        auto mat = PyArrayToCvMat(data);
        vision::SegmentationResult res;
        self.Predict(&mat, &res);
        return res;
      });
}
}  // namespace fastdeploy
