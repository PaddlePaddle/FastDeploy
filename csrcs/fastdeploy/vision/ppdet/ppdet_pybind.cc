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
void BindPPDet(pybind11::module& m) {
  auto ppdet_module =
      m.def_submodule("ppdet", "Module to deploy PaddleDetection.");
  pybind11::class_<vision::ppdet::PPYOLOE, FastDeployModel>(ppdet_module,
                                                            "PPYOLOE")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict", [](vision::ppdet::PPYOLOE& self, pybind11::array& data) {
        auto mat = PyArrayToCvMat(data);
        vision::DetectionResult res;
        self.Predict(&mat, &res);
        return res;
      });
  pybind11::class_<vision::ppdet::PPYOLO, vision::ppdet::PPYOLOE>(ppdet_module,
                                                                  "PPYOLO")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>());
  pybind11::class_<vision::ppdet::PicoDet, vision::ppdet::PPYOLOE>(ppdet_module,
                                                                   "PicoDet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>());
  pybind11::class_<vision::ppdet::YOLOX, vision::ppdet::PPYOLOE>(ppdet_module,
                                                                 "YOLOX")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>());
  pybind11::class_<vision::ppdet::FasterRCNN, vision::ppdet::PPYOLOE>(
      ppdet_module, "FasterRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>());
}
}  // namespace fastdeploy
