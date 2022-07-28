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
void BindBiubug6(pybind11::module& m) {
  auto biubug6_module = m.def_submodule(
      "biubug6", "https://github.com/biubug6/Pytorch_Retinaface");
  pybind11::class_<vision::biubug6::RetinaFace, FastDeployModel>(biubug6_module,
                                                                 "RetinaFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::biubug6::RetinaFace& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceDetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::biubug6::RetinaFace::size)
      .def_readwrite("variance", &vision::biubug6::RetinaFace::variance)
      .def_readwrite("downsample_strides",
                     &vision::biubug6::RetinaFace::downsample_strides)
      .def_readwrite("min_sizes", &vision::biubug6::RetinaFace::min_sizes)
      .def_readwrite("landmarks_per_face",
                     &vision::biubug6::RetinaFace::landmarks_per_face);
}
}  // namespace fastdeploy
