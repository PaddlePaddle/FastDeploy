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
void BindDeepCam(pybind11::module& m) {
  auto deepcam_module =
      m.def_submodule("deepcam", "https://github.com/deepcam-cn/yolov5-face");
  pybind11::class_<vision::deepcam::YOLOv5Face, FastDeployModel>(deepcam_module,
                                                                 "YOLOv5Face")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepcam::YOLOv5Face& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceDetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::deepcam::YOLOv5Face::size)
      .def_readwrite("padding_value",
                     &vision::deepcam::YOLOv5Face::padding_value)
      .def_readwrite("is_mini_pad", &vision::deepcam::YOLOv5Face::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::deepcam::YOLOv5Face::is_no_pad)
      .def_readwrite("is_scale_up", &vision::deepcam::YOLOv5Face::is_scale_up)
      .def_readwrite("stride", &vision::deepcam::YOLOv5Face::stride)
      .def_readwrite("landmarks_per_face",
                     &vision::deepcam::YOLOv5Face::landmarks_per_face);
}

}  // namespace fastdeploy
