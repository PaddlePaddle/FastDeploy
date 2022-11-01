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
void BindPPSeg(pybind11::module& m) {
  pybind11::class_<vision::segmentation::PaddleSegModel, FastDeployModel>(
      m, "PaddleSegModel")
      .def(pybind11::init<std::string, std::string, std::string, RuntimeOption,
                          ModelFormat>())
      .def("predict",
           [](vision::segmentation::PaddleSegModel& self,
              pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             vision::SegmentationResult* res = new vision::SegmentationResult();
             self.Predict(&mat, res);
             return res;
           });
}

void BindPPSegPreprocessor(pybind11::module& m) {
  pybind11::class_<vision::segmentation::PaddleSegPreprocessor>(
      m, "PaddleSegPreprocessor")
      .def(pybind11::init<std::string>())
      .def("run",
           [](vision::segmentation::PaddleSegPreprocessor& self,
              pybind11::array& data) {
             auto mat = PyArrayToCvMat(data);
             fastdeploy::vision::Mat fd_mat(mat);
             FDTensor output;
             std::map<std::string, std::array<int, 2>> im_info;
             im_info["input_shape"] = {static_cast<int>(fd_mat.Height()),
                                       static_cast<int>(fd_mat.Width())};
             self.Run(&fd_mat, &output);
             return make_pair(TensorToPyArray(output), im_info);;
           })
      .def_readwrite("is_vertical_screen",
                     &vision::segmentation::PaddleSegPreprocessor::is_vertical_screen)
      
      .def_readwrite("is_with_softmax",
                     &vision::segmentation::PaddleSegPreprocessor::is_with_softmax)
      
      .def_readwrite("is_with_argmax",
                     &vision::segmentation::PaddleSegPreprocessor::is_with_argmax);
}

void BindPPSegPostprocessor(pybind11::module& m) {
  pybind11::class_<vision::segmentation::PaddleSegPostprocessor>(
      m, "PaddleSegPostprocessor")
      .def(pybind11::init<>())
      .def("run",
           [](vision::segmentation::PaddleSegPostprocessor& self,
              std::vector<pybind11::array> infer_result,
              const std::map<std::string, std::array<int, 2>>& im_info) {
             std::vector<FDTensor> fd_infer_result(infer_result.size());
             PyArrayToTensorList(infer_result, &fd_infer_result, true);
             vision::SegmentationResult* res = new vision::SegmentationResult();
             self.Run(fd_infer_result, res, im_info);
             return res;
           })
      .def("disable_normalize_and_permute",&vision::segmentation::PaddleSegModel::DisableNormalizeAndPermute)
      .def_readwrite("apply_softmax",
                    &vision::segmentation::PaddleSegPostprocessor::apply_softmax)

      .def_readwrite("is_with_softmax",
                     &vision::segmentation::PaddleSegPostprocessor::is_with_softmax)
      
      .def_readwrite("is_with_argmax",
                     &vision::segmentation::PaddleSegPostprocessor::is_with_argmax);
}
}  // namespace fastdeploy
