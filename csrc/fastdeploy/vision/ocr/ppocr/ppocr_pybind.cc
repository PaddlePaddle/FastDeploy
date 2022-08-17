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
void BindPPOCR(pybind11::module& m) {
  // DBDetector
  pybind11::class_<vision::ppocr::DBDetector, FastDeployModel>(m, "DBDetector")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())

      .def_readwrite("max_side_len", &vision::ppocr::DBDetector::max_side_len)
      .def_readwrite("det_db_thresh", &vision::ppocr::DBDetector::det_db_thresh)
      .def_readwrite("det_db_box_thresh",
                     &vision::ppocr::DBDetector::det_db_box_thresh)
      .def_readwrite("det_db_unclip_ratio",
                     &vision::ppocr::DBDetector::det_db_unclip_ratio)
      .def_readwrite("det_db_score_mode",
                     &vision::ppocr::DBDetector::det_db_score_mode)
      .def_readwrite("use_dilation", &vision::ppocr::DBDetector::use_dilation)
      .def_readwrite("mean", &vision::ppocr::DBDetector::mean)
      .def_readwrite("scale", &vision::ppocr::DBDetector::scale)
      .def_readwrite("is_scale", &vision::ppocr::DBDetector::is_scale);

  // Classifier
  pybind11::class_<vision::ppocr::Classifier, FastDeployModel>(m, "Classifier")
      .def(pybind11::init<std::string, std::string, RuntimeOption, Frontend>())

      .def_readwrite("cls_thresh", &vision::ppocr::Classifier::cls_thresh)
      .def_readwrite("cls_batch_num",
                     &vision::ppocr::Classifier::cls_batch_num);

  // Recognizer
  pybind11::class_<vision::ppocr::Recognizer, FastDeployModel>(m, "Recognizer")

      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          Frontend>())

      .def_readwrite("rec_img_h", &vision::ppocr::Recognizer::rec_img_h)
      .def_readwrite("rec_img_w", &vision::ppocr::Recognizer::rec_img_w)
      .def_readwrite("rec_batch_num",
                     &vision::ppocr::Recognizer::rec_batch_num);

  // OCRSys
  pybind11::class_<application::ocrsystem::PPOCRSystemv3, FastDeployModel>(
      m, "PPOCRSystem")

      .def(pybind11::init<fastdeploy::vision::ppocr::DBDetector*,
                          fastdeploy::vision::ppocr::Classifier*,
                          fastdeploy::vision::ppocr::Recognizer*>())

      .def("predict", [](application::ocrsystem::PPOCRSystemv3& self,
                         std::vector<pybind11::array>& data_list) {

        std::vector<cv::Mat> img_list;

        for (int i = 0; i < data_list.size(); i++) {
          auto mat = PyArrayToCvMat(data_list[i]);
          img_list.push_back(mat);
        }

        std::vector<std::vector<vision::OCRResult>> res(self.Predict(img_list));

        return res;
      });
}
}  // namespace fastdeploy
