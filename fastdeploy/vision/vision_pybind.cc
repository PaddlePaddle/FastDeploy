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

void BindPpClsModel(pybind11::module& m);
void BindWongkinyiu(pybind11::module& m);
void BindUltralytics(pybind11::module& m);
void BindMeituan(pybind11::module& m);
void BindMegvii(pybind11::module& m);
#ifdef ENABLE_VISION_VISUALIZE
void BindVisualize(pybind11::module& m);
#endif

void BindVision(pybind11::module& m) {
  pybind11::class_<vision::ClassifyResult>(m, "ClassifyResult")
      .def(pybind11::init())
      .def_readwrite("label_ids", &vision::ClassifyResult::label_ids)
      .def_readwrite("scores", &vision::ClassifyResult::scores)
      .def("__repr__", &vision::ClassifyResult::Str)
      .def("__str__", &vision::ClassifyResult::Str);

  pybind11::class_<vision::DetectionResult>(m, "DetectionResult")
      .def(pybind11::init())
      .def_readwrite("boxes", &vision::DetectionResult::boxes)
      .def_readwrite("scores", &vision::DetectionResult::scores)
      .def_readwrite("label_ids", &vision::DetectionResult::label_ids)
      .def("__repr__", &vision::DetectionResult::Str)
      .def("__str__", &vision::DetectionResult::Str);

  BindPpClsModel(m);
  BindUltralytics(m);
  BindWongkinyiu(m);
  BindMeituan(m);
  BindMegvii(m);
#ifdef ENABLE_VISION_VISUALIZE
  BindVisualize(m);
#endif  
}
} // namespace fastdeploy
