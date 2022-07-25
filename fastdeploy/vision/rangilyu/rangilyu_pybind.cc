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
void BindRangiLyu(pybind11::module& m) {
  auto rangilyu_module =
      m.def_submodule("rangilyu", "https://github.com/RangiLyu/nanodet");
  pybind11::class_<vision::rangilyu::NanoDetPlus, FastDeployModel>(
      rangilyu_module, "NanoDetPlus")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::rangilyu::NanoDetPlus& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::rangilyu::NanoDetPlus::size)
      .def_readwrite("padding_value",
                     &vision::rangilyu::NanoDetPlus::padding_value)
      .def_readwrite("keep_ratio", &vision::rangilyu::NanoDetPlus::keep_ratio)
      .def_readwrite("downsample_strides",
                     &vision::rangilyu::NanoDetPlus::downsample_strides)
      .def_readwrite("max_wh", &vision::rangilyu::NanoDetPlus::max_wh)
      .def_readwrite("reg_max", &vision::rangilyu::NanoDetPlus::reg_max);
}
}  // namespace fastdeploy
