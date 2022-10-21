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

  cv::Mat mat(35, 35, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);
//  vision::Cast::Run(&mat_opencv, "float", vision::ProcLib::OPENCV);
//  vision::Cast::Run(&mat_flycv, "float", vision::ProcLib::FLYCV);
  vision::LimitByStride::Run(&mat_opencv, 32, 1, vision::ProcLib::OPENCV);
  vision::LimitByStride::Run(&mat_flycv, 32, 1, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_type(opencv.dtype, flycv.dtype);

  uint8_t* ptr0 = static_cast<uint8_t*>(mat_opencv.Data());
  uint8_t* ptr1 = static_cast<uint8_t*>(mat_flycv.Data());
  for (int i = 0; i < 32 * 32 * 3; ++i) {
    if (fabs(ptr0[i] - ptr1[i]) > 0) {
      std::cout << "======= " << i << " " << int(ptr0[i]) << " " << int(ptr1[i]) << std::endl;
      ASSERT_EQ(1, 0);
      break;
    }
  }


//  std::vector<uint8_t> data0(35 * 35 * 3, 0);
//  std::vector<uint8_t> data1(35 * 35 * 3, 0);
//  for (size_t i = 0; i < 35 * 35 * 3; ++i) {
//    data0[i] = static_cast<uint8_t>(rand() % 255);
//    data1[i] = data0[i];
//  }
//
//  fcv::Mat mat_flycv(35, 35, fcv::FCVImageType::PACKAGE_BGR_U8, data0.data());
//  cv::Mat mat_opencv(35, 35, CV_8UC3, data1.data());
//
//  cv::Mat new_im0;
//  cv::resize(mat_opencv, new_im0, cv::Size(32, 32), 0, 0, 1);
//
//  fcv::Mat new_im1;
//  fcv::resize(mat_flycv, new_im1, fcv::Size(32, 32), 0, 0, fcv::InterpolationType::INTER_LINEAR);
//
//  uint8_t* ptr0 = static_cast<uint8_t*>(new_im0.ptr());
//  uint8_t* ptr1 = static_cast<uint8_t*>(new_im1.data());
//  for (int i = 0; i < 32 * 32 * 3; ++i) {
//    if (fabs(ptr0[i] - ptr1[i]) > 1) {
//      std::cout << "======= " << i << " " << int(ptr0[i]) << " " << int(ptr1[i]) << std::endl;
//      ASSERT_EQ(1, 0);
//      break;
//    }
//  }
//  check_data(reinterpret_cast<const uint8_t*>(mat_flycv.data()), reinterpret_cast<const uint8_t*>(mat_opencv.ptr()), 35 * 35 * 3);
//  check_data(reinterpret_cast<const uint8_t*>(new_im1.data()), reinterpret_cast<const uint8_t*>(new_im0.ptr()), 32 * 32 * 3);

//  FDTensor opencv;
//  FDTensor flycv;
//
//  mat_opencv.ShareWithTensor(&opencv);
//  mat_flycv.ShareWithTensor(&flycv);
//
//  check_shape(opencv.shape, flycv.shape);
////  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel());
//  check_type(opencv.dtype, flycv.dtype);
}
#endif

}  // namespace fastdeploy
