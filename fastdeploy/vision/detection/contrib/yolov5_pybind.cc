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
void BindYOLOv5(pybind11::module& m) {
  pybind11::class_<vision::detection::YOLOv5, FastDeployModel>(m, "YOLOv5")
      .def(pybind11::init<std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::detection::YOLOv5& self, pybind11::array& data,
              float conf_threshold, float nms_iou_threshold) {
             auto mat = PyArrayToCvMat(data);
             vision::DetectionResult res;
             self.Predict(&mat, &res, conf_threshold, nms_iou_threshold);
             return res;
           })
      .def_static("preprocess",
                  [](pybind11::array& data, const std::vector<int>& size,
                     const std::vector<float> padding_value, bool is_mini_pad,
                     bool is_no_pad, bool is_scale_up, int stride, float max_wh,
                     bool multi_label) {
                    auto mat = PyArrayToCvMat(data);
                    fastdeploy::vision::Mat fd_mat(mat);
                    FDTensor output;
                    std::map<std::string, std::array<float, 2>> im_info;
                    vision::detection::YOLOv5::Preprocess(
                        &fd_mat, &output, &im_info, size, padding_value,
                        is_mini_pad, is_no_pad, is_scale_up, stride, max_wh,
                        multi_label);
                    return make_pair(TensorToPyArray(output), im_info);
                  })
      .def_static(
          "postprocess",
          [](std::vector<pybind11::array> infer_results,
             const std::map<std::string, std::array<float, 2>>& im_info,
             float conf_threshold, float nms_iou_threshold, bool multi_label,
             float max_wh) {
            std::vector<FDTensor> fd_infer_results(infer_results.size());
            PyArrayToTensorList(infer_results, &fd_infer_results, true);
            vision::DetectionResult result;
            vision::detection::YOLOv5::Postprocess(
                fd_infer_results, &result, im_info, conf_threshold,
                nms_iou_threshold, multi_label, max_wh);
            return result;
          })
      .def_readwrite("size", &vision::detection::YOLOv5::size_)
      .def_readwrite("padding_value",
                     &vision::detection::YOLOv5::padding_value_)
      .def_readwrite("is_mini_pad", &vision::detection::YOLOv5::is_mini_pad_)
      .def_readwrite("is_no_pad", &vision::detection::YOLOv5::is_no_pad_)
      .def_readwrite("is_scale_up", &vision::detection::YOLOv5::is_scale_up_)
      .def_readwrite("stride", &vision::detection::YOLOv5::stride_)
      .def_readwrite("max_wh", &vision::detection::YOLOv5::max_wh_)
      .def_readwrite("multi_label", &vision::detection::YOLOv5::multi_label_);
}
}  // namespace fastdeploy
