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
  pybind11::class_<vision::detection::PPYOLOE, FastDeployModel>(m, "PPYOLOE")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PPYOLOE& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::PPYOLO, FastDeployModel>(m, "PPYOLO")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PPYOLO& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::PPYOLOv2, FastDeployModel>(m, "PPYOLOv2")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PPYOLOv2& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::PicoDet, FastDeployModel>(m, "PicoDet")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PicoDet& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::PaddleYOLOX, FastDeployModel>(
      m, "PaddleYOLOX")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PaddleYOLOX& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::FasterRCNN, FastDeployModel>(m,
                                                                   "FasterRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::FasterRCNN& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::YOLOv3, FastDeployModel>(m, "YOLOv3")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::YOLOv3& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::MaskRCNN, FastDeployModel>(m, "MaskRCNN")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::MaskRCNN& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res);
             return res;
           });

  pybind11::class_<vision::detection::PPTinyPose, FastDeployModel>(m,
                                                                   "PPTinyPose")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def("predict",
           [](vision::detection::PPTinyPose& self, pybind11::array& data,
              vision::DetectionResult* detection_result = nullptr) {
             auto mat = PyArrayToCvMat(data);
             vision::KeyPointDetectionResult res;
             self.Predict(&mat, &res, detection_result);
             return res;
           })
      .def_readwrite("use_dark", &vision::detection::PPTinyPose::use_dark);
}
}  // namespace fastdeploy
