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
void BindDeepInsight(pybind11::module& m) {
  auto deepinsight_module =
      m.def_submodule("deepinsight", "https://github.com/deepinsight");
  // Bind SCRFD
  pybind11::class_<vision::deepinsight::SCRFD, FastDeployModel>(
      deepinsight_module, "SCRFD")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::SCRFD& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceDetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_readwrite("size", &vision::deepinsight::SCRFD::size)
      .def_readwrite("padding_value",
                     &vision::deepinsight::SCRFD::padding_value)
      .def_readwrite("is_mini_pad", &vision::deepinsight::SCRFD::is_mini_pad)
      .def_readwrite("is_no_pad", &vision::deepinsight::SCRFD::is_no_pad)
      .def_readwrite("is_scale_up", &vision::deepinsight::SCRFD::is_scale_up)
      .def_readwrite("stride", &vision::deepinsight::SCRFD::stride)
      .def_readwrite("use_kps", &vision::deepinsight::SCRFD::use_kps)
      .def_readwrite("max_nms", &vision::deepinsight::SCRFD::max_nms)
      .def_readwrite("downsample_strides",
                     &vision::deepinsight::SCRFD::downsample_strides)
      .def_readwrite("num_anchors", &vision::deepinsight::SCRFD::num_anchors)
      .def_readwrite("landmarks_per_face",
                     &vision::deepinsight::SCRFD::landmarks_per_face);
  // Bind InsightFaceRecognitionModel
  pybind11::class_<vision::deepinsight::InsightFaceRecognitionModel,
                   FastDeployModel>(deepinsight_module,
                                    "InsightFaceRecognitionModel")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::InsightFaceRecognitionModel& self,
              pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size",
                     &vision::deepinsight::InsightFaceRecognitionModel::size)
      .def_readwrite("alpha",
                     &vision::deepinsight::InsightFaceRecognitionModel::alpha)
      .def_readwrite("beta",
                     &vision::deepinsight::InsightFaceRecognitionModel::beta)
      .def_readwrite("swap_rb",
                     &vision::deepinsight::InsightFaceRecognitionModel::swap_rb)
      .def_readwrite(
          "l2_normalize",
          &vision::deepinsight::InsightFaceRecognitionModel::l2_normalize);
  // Bind ArcFace
  pybind11::class_<vision::deepinsight::ArcFace,
                   vision::deepinsight::InsightFaceRecognitionModel>(
      deepinsight_module, "ArcFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::ArcFace& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::deepinsight::ArcFace::size)
      .def_readwrite("alpha", &vision::deepinsight::ArcFace::alpha)
      .def_readwrite("beta", &vision::deepinsight::ArcFace::beta)
      .def_readwrite("swap_rb", &vision::deepinsight::ArcFace::swap_rb)
      .def_readwrite("l2_normalize",
                     &vision::deepinsight::ArcFace::l2_normalize);
  // Bind CosFace
  pybind11::class_<vision::deepinsight::CosFace,
                   vision::deepinsight::InsightFaceRecognitionModel>(
      deepinsight_module, "CosFace")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::CosFace& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::deepinsight::CosFace::size)
      .def_readwrite("alpha", &vision::deepinsight::CosFace::alpha)
      .def_readwrite("beta", &vision::deepinsight::CosFace::beta)
      .def_readwrite("swap_rb", &vision::deepinsight::CosFace::swap_rb)
      .def_readwrite("l2_normalize",
                     &vision::deepinsight::CosFace::l2_normalize);
  // Bind Partial FC
  pybind11::class_<vision::deepinsight::PartialFC,
                   vision::deepinsight::InsightFaceRecognitionModel>(
      deepinsight_module, "PartialFC")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::PartialFC& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::deepinsight::PartialFC::size)
      .def_readwrite("alpha", &vision::deepinsight::PartialFC::alpha)
      .def_readwrite("beta", &vision::deepinsight::PartialFC::beta)
      .def_readwrite("swap_rb", &vision::deepinsight::PartialFC::swap_rb)
      .def_readwrite("l2_normalize",
                     &vision::deepinsight::PartialFC::l2_normalize);
  // Bind VPL
  pybind11::class_<vision::deepinsight::VPL,
                   vision::deepinsight::InsightFaceRecognitionModel>(
      deepinsight_module, "VPL")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def("predict",
           [](vision::deepinsight::VPL& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::FaceRecognitionResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def_readwrite("size", &vision::deepinsight::VPL::size)
      .def_readwrite("alpha", &vision::deepinsight::VPL::alpha)
      .def_readwrite("beta", &vision::deepinsight::VPL::beta)
      .def_readwrite("swap_rb", &vision::deepinsight::VPL::swap_rb)
      .def_readwrite("l2_normalize", &vision::deepinsight::VPL::l2_normalize);
}

}  // namespace fastdeploy
