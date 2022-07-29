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
void BindPpogg(pybind11::module& m) {
  auto ppogg_module =
      m.def_submodule("ppogg", "https://github.com/ppogg/YOLOv5-Lite");
  pybind11::class_<vision::ppogg::YOLOv5Lite, FastDeployModel>(ppogg_module,
                                                               "YOLOv5Lite")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::ppogg::YOLOv5Lite& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::ppogg::YOLOv5Lite::size)
      .def_readwrite("padding_value", &vision::ppogg::YOLOv5Lite::padding_value)
      .def_readwrite("is_mini_pad", &vision::ppogg::YOLOv5Lite::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::ppogg::YOLOv5Lite::is_no_pad)
      .def_readwrite("is_scale_up", &vision::ppogg::YOLOv5Lite::is_scale_up)
      .def_readwrite("stride", &vision::ppogg::YOLOv5Lite::stride)
      .def_readwrite("max_wh", &vision::ppogg::YOLOv5Lite::max_wh)
      .def_readwrite("anchor_config", &vision::ppogg::YOLOv5Lite::anchor_config)
      .def_readwrite("is_decode_exported",
                     &vision::ppogg::YOLOv5Lite::is_decode_exported);
}
}  // namespace fastdeploy
