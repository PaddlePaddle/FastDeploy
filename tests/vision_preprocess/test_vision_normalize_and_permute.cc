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
TEST(fastdeploy, falconcv_norm_and_perm0) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();
  cv::Mat mat2 = mat.clone();

  std::vector<float> mean({0.25, 0.35, 0.45});
  std::vector<float> std({0.33, 0.22, 0.54});
  std::vector<float> min;
  std::vector<float> max;

  vision::Mat mat_opencv(mat);
  vision::Mat mat_opencv1(mat);
  vision::Mat mat_falconcv(mat1);

  vision::Normalize::Run(&mat_opencv, mean, std, true, min, max, vision::ProcLib::OPENCV);
  vision::HWC2CHW::Run(&mat_opencv, vision::ProcLib::OPENCV);
  vision::NormalizeAndPermute::Run(&mat_opencv1, mean, std, true, min, max, vision::ProcLib::OPENCV);

  vision::NormalizeAndPermute::Run(&mat_falconcv, mean, std, true, min, max, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor opencv1;
  FDTensor falconcv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_opencv1.ShareWithTensor(&opencv1);
  mat_falconcv.ShareWithTensor(&falconcv);

  check_shape(opencv.shape, falconcv.shape);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(falconcv.Data()), opencv.Numel());
  check_data(reinterpret_cast<const float*>(opencv1.Data()), reinterpret_cast<const float*>(falconcv.Data()), opencv1.Numel());
  check_type(opencv.dtype, falconcv.dtype);
}
#endif

}  // namespace fastdeploy
