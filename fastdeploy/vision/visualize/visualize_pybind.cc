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
void BindVisualize(pybind11::module& m) {
  pybind11::class_<vision::Visualize>(m, "Visualize")
      .def(pybind11::init<>())
      .def_static("vis_detection", [](pybind11::array& im_data,
                                      vision::DetectionResult& result,
                                      int line_size, float font_size) {
        auto im = PyArrayToCvMat(im_data);
        vision::Visualize::VisDetection(&im, result, line_size, font_size);
      });
}
} // namespace fastdeploy
