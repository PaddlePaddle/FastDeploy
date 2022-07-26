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
void BindMegvii(pybind11::module& m) {
  auto megvii_module =
      m.def_submodule("megvii", "https://github.com/megvii/YOLOX");
  pybind11::class_<vision::megvii::YOLOX, FastDeployModel>(
      megvii_module, "YOLOX")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::megvii::YOLOX& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::megvii::YOLOX::size)
      .def_readwrite("padding_value",
                     &vision::megvii::YOLOX::padding_value)
      .def_readwrite("is_decode_exported", 
                     &vision::megvii::YOLOX::is_decode_exported)
      .def_readwrite("downsample_strides",
                     &vision::megvii::YOLOX::downsample_strides)
      .def_readwrite("max_wh", &vision::megvii::YOLOX::max_wh);
}
}  // namespace fastdeploy
