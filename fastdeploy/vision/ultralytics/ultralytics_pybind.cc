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
void BindUltralytics(pybind11::module& m) {
  auto ultralytics_module =
      m.def_submodule("ultralytics", "https://github.com/ultralytics/yolov5");
  pybind11::class_<vision::ultralytics::YOLOv5, FastDeployModel>(
      ultralytics_module, "YOLOv5")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::ultralytics::YOLOv5& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::ultralytics::YOLOv5::size)
      .def_readwrite("padding_value",
                     &vision::ultralytics::YOLOv5::padding_value)
      .def_readwrite("is_mini_pad", &vision::ultralytics::YOLOv5::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::ultralytics::YOLOv5::is_no_pad)
      .def_readwrite("is_scale_up", &vision::ultralytics::YOLOv5::is_scale_up)
      .def_readwrite("stride", &vision::ultralytics::YOLOv5::stride)
      .def_readwrite("max_wh", &vision::ultralytics::YOLOv5::max_wh)
      .def_readwrite("multi_label", &vision::ultralytics::YOLOv5::multi_label);
}
}  // namespace fastdeploy
