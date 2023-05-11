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
void BindPPOCRv4(pybind11::module& m) {
  // PPOCRv4
  pybind11::class_<pipeline::PPOCRv4, FastDeployModel>(m, "PPOCRv4")

      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Classifier*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def_property("cls_batch_size", &pipeline::PPOCRv4::GetClsBatchSize,
                    &pipeline::PPOCRv4::SetClsBatchSize)
      .def_property("rec_batch_size", &pipeline::PPOCRv4::GetRecBatchSize,
                    &pipeline::PPOCRv4::SetRecBatchSize)
      .def("clone", [](pipeline::PPOCRv4& self) { return self.Clone(); })
      .def("predict",
           [](pipeline::PPOCRv4& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict",
           [](pipeline::PPOCRv4& self, std::vector<pybind11::array>& data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::OCRResult> results;
             self.BatchPredict(images, &results);
             return results;
           });
}
void BindPPOCRv3(pybind11::module& m) {
  // PPOCRv3
  pybind11::class_<pipeline::PPOCRv3, FastDeployModel>(m, "PPOCRv3")

      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Classifier*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def_property("cls_batch_size", &pipeline::PPOCRv3::GetClsBatchSize,
                    &pipeline::PPOCRv3::SetClsBatchSize)
      .def_property("rec_batch_size", &pipeline::PPOCRv3::GetRecBatchSize,
                    &pipeline::PPOCRv3::SetRecBatchSize)
      .def("clone", [](pipeline::PPOCRv3& self) { return self.Clone(); })
      .def("predict",
           [](pipeline::PPOCRv3& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict",
           [](pipeline::PPOCRv3& self, std::vector<pybind11::array>& data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::OCRResult> results;
             self.BatchPredict(images, &results);
             return results;
           });
}

void BindPPOCRv2(pybind11::module& m) {
  // PPOCRv2
  pybind11::class_<pipeline::PPOCRv2, FastDeployModel>(m, "PPOCRv2")
      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Classifier*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Recognizer*>())
      .def_property("cls_batch_size", &pipeline::PPOCRv2::GetClsBatchSize,
                    &pipeline::PPOCRv2::SetClsBatchSize)
      .def_property("rec_batch_size", &pipeline::PPOCRv2::GetRecBatchSize,
                    &pipeline::PPOCRv2::SetRecBatchSize)
      .def("clone", [](pipeline::PPOCRv2& self) { return self.Clone(); })
      .def("predict",
           [](pipeline::PPOCRv2& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict",
           [](pipeline::PPOCRv2& self, std::vector<pybind11::array>& data) {
             std::vector<cv::Mat> images;
             for (size_t i = 0; i < data.size(); ++i) {
               images.push_back(PyArrayToCvMat(data[i]));
             }
             std::vector<vision::OCRResult> results;
             self.BatchPredict(images, &results);
             return results;
           });
}

void BindPPStructureV2Table(pybind11::module& m) {
  // PPStructureV2Table
  pybind11::class_<pipeline::PPStructureV2Table, FastDeployModel>(
      m, "PPStructureV2Table")
      .def(pybind11::init<fastdeploy::vision::ocr::DBDetector*,
                          fastdeploy::vision::ocr::Recognizer*,
                          fastdeploy::vision::ocr::StructureV2Table*>())
      .def_property("rec_batch_size",
                    &pipeline::PPStructureV2Table::GetRecBatchSize,
                    &pipeline::PPStructureV2Table::SetRecBatchSize)
      .def("clone",
           [](pipeline::PPStructureV2Table& self) { return self.Clone(); })
      .def("predict",
           [](pipeline::PPStructureV2Table& self, pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::OCRResult res;
             self.Predict(&mat, &res);
             return res;
           })
      .def("batch_predict", [](pipeline::PPStructureV2Table& self,
                               std::vector<pybind11::array>& data) {
        std::vector<cv::Mat> images;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        std::vector<vision::OCRResult> results;
        self.BatchPredict(images, &results);
        return results;
      });
}

}  // namespace fastdeploy
