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
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_static("preprocess",
                  [](pybind11::array& data) {
                    auto mat = PyArrayToCvMat(data);
                    fastdeploy::vision::Mat fd_mat(mat);
                    FDTensor output;
                    std::map<std::string, std::array<float, 2>> im_info;
                    vision::ocr::DBDetector::Preprocess(
                        &fd_mat, &output, &im_info);
                    return make_pair(TensorToPyArray(output), im_info);
                  })
      .def_static(
          "postprocess",
          [](std::vector<pybind11::array> infer_results,
             const std::map<std::string, std::array<float, 2>>& im_info) {
            std::vector<FDTensor> fd_infer_results(infer_results.size());
            PyArrayToTensorList(infer_results, &fd_infer_results, true);
            std::vector<std::array<int, 8>> boxes_result;
            vision::ocr::DBDetector::Postprocess(
                fd_infer_results, &boxes_result, im_info);
            return boxes_result;
          })
        
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
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_static("preprocess",
                  [](pybind11::array& data) {
                    auto mat = PyArrayToCvMat(data);
                    fastdeploy::vision::Mat fd_mat(mat);
                    FDTensor output;
                    vision::ocr::Classifier::Preprocess(
                        &fd_mat, &output);
                    return TensorToPyArray(output);
                  })
      .def_static(
          "postprocess",
          [](std::vector<pybind11::array> infer_results) {
            std::vector<FDTensor> fd_infer_results(infer_results.size());
            PyArrayToTensorList(infer_results, &fd_infer_results, true);
            std::tuple<int, float> result;
            vision::ocr::Classifier::Postprocess(
                fd_infer_results, &result);
            return result;
          })
      .def_readwrite("cls_thresh", &vision::ocr::Classifier::cls_thresh)
      .def_readwrite("cls_image_shape",
                     &vision::ocr::Classifier::cls_image_shape)
      .def_readwrite("cls_batch_num", &vision::ocr::Classifier::cls_batch_num);

  // Recognizer
  pybind11::class_<vision::ocr::Recognizer, FastDeployModel>(m, "Recognizer")

      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def(pybind11::init<>())
      .def_static("preprocess",
                  [](pybind11::array& data,
                     const std::vector<int>& rec_image_shape) {
                    auto mat = PyArrayToCvMat(data);
                    fastdeploy::vision::Mat fd_mat(mat);
                    FDTensor output;
                    vision::ocr::Recognizer::Preprocess(
                        &fd_mat, &output, rec_image_shape);
                    return TensorToPyArray(output);
                  })
      .def_static(
          "postprocess",
          [](std::vector<pybind11::array> infer_results,
             std::vector<std::string>& label_list) {
            std::vector<FDTensor> fd_infer_results(infer_results.size());
            PyArrayToTensorList(infer_results, &fd_infer_results, true);
            std::tuple<std::string, float> rec_result;
            vision::ocr::Recognizer::Postprocess(
                fd_infer_results, &rec_result, label_list);
            return rec_result;
          })
      .def_readwrite("rec_img_h", &vision::ocr::Recognizer::rec_img_h)
      .def_readwrite("rec_img_w", &vision::ocr::Recognizer::rec_img_w)
      .def_readwrite("rec_batch_num", &vision::ocr::Recognizer::rec_batch_num);
}
}  // namespace fastdeploy
