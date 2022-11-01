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
TEST(fastdeploy, flycv_limit_stride1) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(35, 69, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);
  vision::LimitByStride::Run(&mat_opencv, 38, 1, vision::ProcLib::OPENCV);
  vision::LimitByStride::Run(&mat_flycv, 38, 1, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_type(opencv.dtype, flycv.dtype);
  check_data(reinterpret_cast<const uint8_t*>(opencv.Data()), reinterpret_cast<const uint8_t*>(flycv.Data()), opencv.Numel(), 1);
}

TEST(fastdeploy, flycv_limit_stride2) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(35, 69, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);
  vision::Cast::Run(&mat_opencv, "float", vision::ProcLib::OPENCV);
  vision::Cast::Run(&mat_flycv, "float", vision::ProcLib::FLYCV);
  vision::LimitByStride::Run(&mat_opencv, 32, 1, vision::ProcLib::OPENCV);
  vision::LimitByStride::Run(&mat_flycv, 32, 1, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_type(opencv.dtype, flycv.dtype);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel(), 0, 0);
}

//TEST(fastdeploy, flycv_limit_stride3) {
//  CheckShape check_shape;
//  CheckData check_data;
//  CheckType check_type;
//
//  cv::Mat mat(35, 69, CV_8UC3);
//  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
//  cv::Mat mat1 = mat.clone();
//
//  vision::Mat mat_opencv(mat);
//  vision::Mat mat_flycv(mat1);
////  vision::Cast::Run(&mat_opencv, "float", vision::ProcLib::OPENCV);
////  vision::Cast::Run(&mat_flycv, "float", vision::ProcLib::FLYCV);
//  vision::LimitByStride::Run(&mat_opencv, 32, 0, vision::ProcLib::OPENCV);
//  vision::LimitByStride::Run(&mat_flycv, 32, 0, vision::ProcLib::FLYCV);
//
//  FDTensor opencv;
//  FDTensor flycv;
//
//  mat_opencv.ShareWithTensor(&opencv);
//  mat_flycv.ShareWithTensor(&flycv);
//
//  check_shape(opencv.shape, flycv.shape);
//  check_type(opencv.dtype, flycv.dtype);
//  check_data(reinterpret_cast<const uint8_t*>(opencv.Data()), reinterpret_cast<const uint8_t*>(flycv.Data()), opencv.Numel(), 1);
//}

//TEST(fastdeploy, flycv_limit_stride4) {
//  CheckShape check_shape;
//  CheckData check_data;
//  CheckType check_type;
//
//  cv::Mat mat(35, 69, CV_8UC3);
//  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
//  cv::Mat mat1 = mat.clone();
//
//  vision::Mat mat_opencv(mat);
//  vision::Mat mat_flycv(mat1);
//  vision::Cast::Run(&mat_opencv, "float", vision::ProcLib::OPENCV);
//  vision::Cast::Run(&mat_flycv, "float", vision::ProcLib::FLYCV);
//  vision::LimitByStride::Run(&mat_opencv, 32, 2, vision::ProcLib::OPENCV);
//  vision::LimitByStride::Run(&mat_flycv, 32, 2, vision::ProcLib::FLYCV);
//
//  FDTensor opencv;
//  FDTensor flycv;
//
//  mat_opencv.ShareWithTensor(&opencv);
//  mat_flycv.ShareWithTensor(&flycv);
//
//  check_shape(opencv.shape, flycv.shape);
//  check_type(opencv.dtype, flycv.dtype);
//  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel(), 0, 0);
//}

#endif

}  // namespace fastdeploy
