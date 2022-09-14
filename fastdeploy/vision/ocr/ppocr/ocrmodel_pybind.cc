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
#include <pybind11/stl.h>
#include "fastdeploy/pybind/main.h"

namespace fastdeploy {
void BindPPOCRModel(pybind11::module& m) {
  // DBDetector
  pybind11::class_<vision::ocr::DBDetector, FastDeployModel>(m, "DBDetector")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def(pybind11::init<>())

      .def_readwrite("max_side_len", &vision::ocr::DBDetector::max_side_len)
      .def_readwrite("det_db_thresh", &vision::ocr::DBDetector::det_db_thresh)
      .def_readwrite("det_db_box_thresh",
                     &vision::ocr::DBDetector::det_db_box_thresh)
      .def_readwrite("det_db_unclip_ratio",
                     &vision::ocr::DBDetector::det_db_unclip_ratio)
      .def_readwrite("det_db_score_mode",
                     &vision::ocr::DBDetector::det_db_score_mode)
      .def_readwrite("use_dilation", &vision::ocr::DBDetector::use_dilation)
      .def_readwrite("mean", &vision::ocr::DBDetector::mean)
      .def_readwrite("scale", &vision::ocr::DBDetector::scale)
      .def_readwrite("is_scale", &vision::ocr::DBDetector::is_scale);

  // Classifier
  pybind11::class_<vision::ocr::Classifier, FastDeployModel>(m, "Classifier")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())
      .def(pybind11::init<>())

      .def_readwrite("cls_thresh", &vision::ocr::Classifier::cls_thresh)
      .def_readwrite("cls_image_shape",
                     &vision::ocr::Classifier::cls_image_shape)
      .def_readwrite("cls_batch_num", &vision::ocr::Classifier::cls_batch_num);

  // Recognizer
  pybind11::class_<vision::ocr::Recognizer, FastDeployModel>(m, "Recognizer")

      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())
      .def(pybind11::init<>())

      .def_readwrite("rec_img_h", &vision::ocr::Recognizer::rec_img_h)
      .def_readwrite("rec_img_w", &vision::ocr::Recognizer::rec_img_w)
      .def_readwrite("rec_batch_num", &vision::ocr::Recognizer::rec_batch_num);
}
}  // namespace fastdeploy
