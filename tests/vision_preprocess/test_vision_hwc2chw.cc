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

#include <array>
#include <vector>
#include "fastdeploy/vision.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "gtest_utils.h"

namespace fastdeploy {

#ifdef ENABLE_FLYCV
TEST(fastdeploy, falconcv_hwc2chw1) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_falconcv(mat1);
  vision::Cast::Run(&mat_opencv, "float", vision::ProcLib::OPENCV);
  vision::Cast::Run(&mat_falconcv, "float", vision::ProcLib::FLYCV);
  vision::HWC2CHW::Run(&mat_opencv, vision::ProcLib::OPENCV);
  vision::HWC2CHW::Run(&mat_falconcv, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor falconcv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_falconcv.ShareWithTensor(&falconcv);

  check_shape(opencv.shape, falconcv.shape);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(falconcv.Data()), opencv.Numel());
  check_type(opencv.dtype, falconcv.dtype);
}

//TEST(fastdeploy, falconcv_hwc2chw2) {
//  CheckShape check_shape;
//  CheckData check_data;
//  CheckType check_type;
//
//  cv::Mat mat(64, 64, CV_8UC3);
//  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
//  cv::Mat mat1 = mat.clone();
//
//  vision::Mat mat_opencv(mat);
//  vision::Mat mat_falconcv(mat1);
//  vision::HWC2CHW::Run(&mat_opencv, vision::ProcLib::OPENCV);
//  vision::HWC2CHW::Run(&mat_falconcv, vision::ProcLib::FLYCV);
//
//  FDTensor opencv;
//  FDTensor falconcv;
//
//  mat_opencv.ShareWithTensor(&opencv);
//  mat_falconcv.ShareWithTensor(&falconcv);
//
//  check_shape(opencv.shape, falconcv.shape);
//  check_data(reinterpret_cast<const uint8_t*>(opencv.Data()), reinterpret_cast<const uint8_t*>(falconcv.Data()), opencv.Numel());
//  check_type(opencv.dtype, falconcv.dtype);
//}
#endif

}  // namespace fastdeploy
