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
  m.def("sort_boxes", [](std::vector<std::array<int, 8>>& boxes) {
       vision::ocr::SortBoxes(&boxes);
       return boxes;
  });
  
  // DBDetector
  pybind11::class_<vision::ocr::DBDetectorPreprocessor>(m, "DBDetectorPreprocessor")
      .def(pybind11::init<>())
      .def_property("max_side_len", &vision::ocr::DBDetectorPreprocessor::GetMaxSideLen, &vision::ocr::DBDetectorPreprocessor::SetMaxSideLen)
      .def_property("mean", &vision::ocr::DBDetectorPreprocessor::GetMean, &vision::ocr::DBDetectorPreprocessor::SetMean)
      .def_property("scale", &vision::ocr::DBDetectorPreprocessor::GetScale, &vision::ocr::DBDetectorPreprocessor::SetScale)
      .def_property("is_scale", &vision::ocr::DBDetectorPreprocessor::GetIsScale, &vision::ocr::DBDetectorPreprocessor::SetIsScale)
      .def("run", [](vision::ocr::DBDetectorPreprocessor& self, std::vector<pybind11::array>& im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        std::vector<std::array<int, 4>> batch_det_img_info;
        self.Run(&images, &outputs, &batch_det_img_info);
        for(size_t i = 0; i< outputs.size(); ++i){
          outputs[i].StopSharing();
        }
        return std::make_pair(outputs, batch_det_img_info);
      });

  pybind11::class_<vision::ocr::DBDetectorPostprocessor>(m, "DBDetectorPostprocessor")
      .def(pybind11::init<>())
      .def_property("det_db_thresh", &vision::ocr::DBDetectorPostprocessor::GetDetDBThresh, &vision::ocr::DBDetectorPostprocessor::SetDetDBThresh) 
      .def_property("det_db_box_thresh", &vision::ocr::DBDetectorPostprocessor::GetDetDBBoxThresh, &vision::ocr::DBDetectorPostprocessor::SetDetDBBoxThresh) 
      .def_property("det_db_unclip_ratio", &vision::ocr::DBDetectorPostprocessor::GetDetDBUnclipRatio, &vision::ocr::DBDetectorPostprocessor::SetDetDBUnclipRatio) 
      .def_property("det_db_score_mode", &vision::ocr::DBDetectorPostprocessor::GetDetDBScoreMode, &vision::ocr::DBDetectorPostprocessor::SetDetDBScoreMode) 
      .def_property("use_dilation", &vision::ocr::DBDetectorPostprocessor::GetUseDilation, &vision::ocr::DBDetectorPostprocessor::SetUseDilation) 

      .def("run", [](vision::ocr::DBDetectorPostprocessor& self,
                     std::vector<FDTensor>& inputs,
                     const std::vector<std::array<int, 4>>& batch_det_img_info) {
        std::vector<std::vector<std::array<int, 8>>> results;

        if (!self.Run(inputs, &results, batch_det_img_info)) {
          throw std::runtime_error("Failed to preprocess the input data in DBDetectorPostprocessor.");
        }
        return results;
      })
      .def("run", [](vision::ocr::DBDetectorPostprocessor& self,
                     std::vector<pybind11::array>& input_array,
                     const std::vector<std::array<int, 4>>& batch_det_img_info) {
        std::vector<std::vector<std::array<int, 8>>> results;
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        if (!self.Run(inputs, &results, batch_det_img_info)) {
          throw std::runtime_error("Failed to preprocess the input data in DBDetectorPostprocessor.");
        }
        return results;
      });

  pybind11::class_<vision::ocr::DBDetector, FastDeployModel>(m, "DBDetector")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor", &vision::ocr::DBDetector::GetPreprocessor)
      .def_property_readonly("postprocessor", &vision::ocr::DBDetector::GetPostprocessor)
      .def("predict", [](vision::ocr::DBDetector& self,
                         pybind11::array& data) {
        auto mat = PyArrayToCvMat(data);
        std::vector<std::array<int, 8>> boxes_result;
        self.Predict(mat, &boxes_result);
        return boxes_result;
      })
      .def("batch_predict", [](vision::ocr::DBDetector& self, std::vector<pybind11::array>& data) {
        std::vector<cv::Mat> images;
        std::vector<std::vector<std::array<int, 8>>> det_results;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        self.BatchPredict(images, &det_results);
        return det_results;
      });

  // Classifier
  pybind11::class_<vision::ocr::ClassifierPreprocessor>(m, "ClassifierPreprocessor")
      .def(pybind11::init<>())
      .def_property("cls_image_shape", &vision::ocr::ClassifierPreprocessor::GetClsImageShape, &vision::ocr::ClassifierPreprocessor::SetClsImageShape)
      .def_property("mean", &vision::ocr::ClassifierPreprocessor::GetMean, &vision::ocr::ClassifierPreprocessor::SetMean)
      .def_property("scale", &vision::ocr::ClassifierPreprocessor::GetScale, &vision::ocr::ClassifierPreprocessor::SetScale)
      .def_property("is_scale", &vision::ocr::ClassifierPreprocessor::GetIsScale, &vision::ocr::ClassifierPreprocessor::SetIsScale)
      .def("run", [](vision::ocr::ClassifierPreprocessor& self, std::vector<pybind11::array>& im_list) {
        std::vector<vision::FDMat> images;
        for (size_t i = 0; i < im_list.size(); ++i) {
          images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
        }
        std::vector<FDTensor> outputs;
        if (!self.Run(&images, &outputs)) {
          throw std::runtime_error("Failed to preprocess the input data in ClassifierPreprocessor.");
        }
        for(size_t i = 0; i< outputs.size(); ++i){
          outputs[i].StopSharing();
        }
        return outputs;
      });

  pybind11::class_<vision::ocr::ClassifierPostprocessor>(m, "ClassifierPostprocessor")
      .def(pybind11::init<>())
      .def_property("cls_thresh", &vision::ocr::ClassifierPostprocessor::GetClsThresh, &vision::ocr::ClassifierPostprocessor::SetClsThresh) 
      .def("run", [](vision::ocr::ClassifierPostprocessor& self,
                     std::vector<FDTensor>& inputs) {
        std::vector<int> cls_labels;
        std::vector<float> cls_scores;
        if (!self.Run(inputs, &cls_labels, &cls_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in ClassifierPostprocessor.");
        }
        return std::make_pair(cls_labels,cls_scores);
      })
      .def("run", [](vision::ocr::ClassifierPostprocessor& self,
                     std::vector<pybind11::array>& input_array) {
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        std::vector<int> cls_labels;
        std::vector<float> cls_scores;
        if (!self.Run(inputs, &cls_labels, &cls_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in ClassifierPostprocessor.");
        }
        return std::make_pair(cls_labels,cls_scores);
      });
  
  pybind11::class_<vision::ocr::Classifier, FastDeployModel>(m, "Classifier")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor", &vision::ocr::Classifier::GetPreprocessor)
      .def_property_readonly("postprocessor", &vision::ocr::Classifier::GetPostprocessor)
      .def("predict", [](vision::ocr::Classifier& self,
                         pybind11::array& data) {
        auto mat = PyArrayToCvMat(data);
        int32_t cls_label;
        float cls_score;
        self.Predict(mat, &cls_label, &cls_score);
        return std::make_pair(cls_label, cls_score);
      })
      .def("batch_predict", [](vision::ocr::Classifier& self, std::vector<pybind11::array>& data) {
        std::vector<cv::Mat> images;
        std::vector<int32_t> cls_labels;
        std::vector<float> cls_scores;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        self.BatchPredict(images, &cls_labels, &cls_scores);
        return std::make_pair(cls_labels, cls_scores);
      });

  // Recognizer
  pybind11::class_<vision::ocr::RecognizerPreprocessor>(m, "RecognizerPreprocessor")
    .def(pybind11::init<>())
    .def_property("static_shape_infer", &vision::ocr::RecognizerPreprocessor::GetStaticShapeInfer, &vision::ocr::RecognizerPreprocessor::SetStaticShapeInfer) 
    .def_property("rec_image_shape", &vision::ocr::RecognizerPreprocessor::GetRecImageShape, &vision::ocr::RecognizerPreprocessor::SetRecImageShape)
    .def_property("mean", &vision::ocr::RecognizerPreprocessor::GetMean, &vision::ocr::RecognizerPreprocessor::SetMean)
    .def_property("scale", &vision::ocr::RecognizerPreprocessor::GetScale, &vision::ocr::RecognizerPreprocessor::SetScale)
    .def_property("is_scale", &vision::ocr::RecognizerPreprocessor::GetIsScale, &vision::ocr::RecognizerPreprocessor::SetIsScale)
    .def("run", [](vision::ocr::RecognizerPreprocessor& self, std::vector<pybind11::array>& im_list) {
      std::vector<vision::FDMat> images;
      for (size_t i = 0; i < im_list.size(); ++i) {
        images.push_back(vision::WrapMat(PyArrayToCvMat(im_list[i])));
      }
      std::vector<FDTensor> outputs;
      if (!self.Run(&images, &outputs)) {
        throw std::runtime_error("Failed to preprocess the input data in RecognizerPreprocessor.");
      }
      for(size_t i = 0; i< outputs.size(); ++i){
        outputs[i].StopSharing();
      }
      return outputs;
    });

  pybind11::class_<vision::ocr::RecognizerPostprocessor>(m, "RecognizerPostprocessor")
      .def(pybind11::init<std::string>())
      .def("run", [](vision::ocr::RecognizerPostprocessor& self,
                     std::vector<FDTensor>& inputs) {
        std::vector<std::string> texts;
        std::vector<float> rec_scores;
        if (!self.Run(inputs, &texts, &rec_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in RecognizerPostprocessor.");
        }
        return std::make_pair(texts, rec_scores);
      })
      .def("run", [](vision::ocr::RecognizerPostprocessor& self,
                     std::vector<pybind11::array>& input_array) {
        std::vector<FDTensor> inputs;
        PyArrayToTensorList(input_array, &inputs, /*share_buffer=*/true);
        std::vector<std::string> texts;
        std::vector<float> rec_scores;
        if (!self.Run(inputs, &texts, &rec_scores)) {
          throw std::runtime_error("Failed to preprocess the input data in RecognizerPostprocessor.");
        }
        return std::make_pair(texts, rec_scores);
      });

  pybind11::class_<vision::ocr::Recognizer, FastDeployModel>(m, "Recognizer")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_property_readonly("preprocessor", &vision::ocr::Recognizer::GetPreprocessor)
      .def_property_readonly("postprocessor", &vision::ocr::Recognizer::GetPostprocessor)
      .def("predict", [](vision::ocr::Recognizer& self,
                         pybind11::array& data) {
        auto mat = PyArrayToCvMat(data);
        std::string text;
        float rec_score;
        self.Predict(mat, &text, &rec_score);
        return std::make_pair(text, rec_score);
      })
      .def("batch_predict", [](vision::ocr::Recognizer& self, std::vector<pybind11::array>& data) {
        std::vector<cv::Mat> images;
        std::vector<std::string> texts;
        std::vector<float> rec_scores;
        for (size_t i = 0; i < data.size(); ++i) {
          images.push_back(PyArrayToCvMat(data[i]));
        }
        self.BatchPredict(images, &texts, &rec_scores);
        return std::make_pair(texts, rec_scores);
      });
}
}  // namespace fastdeploy
