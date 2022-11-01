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

#ifdef ENABLE_OPENCV_CUDA
TEST(fastdeploy, opencv_cuda_norm1) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_opencv_cuda(mat1);

  std::vector<float> mean({0.25, 0.35, 0.45});
  std::vector<float> std({0.33, 0.22, 0.54});
  std::vector<float> min;
  std::vector<float> max;
  vision::Normalize::Run(&mat_opencv, mean, std, true, min, max, vision::ProcLib::OPENCV);
  vision::Normalize::Run(&mat_opencv_cuda, mean, std, true, min, max, vision::ProcLib::OPENCVCUDA);

  FDTensor opencv;
  FDTensor opencv_cuda;

  mat_opencv.ShareWithTensor(&opencv);
  mat_opencv_cuda.ShareWithTensor(&opencv_cuda);

  check_shape(opencv.shape, opencv_cuda.shape);
  check_type(opencv.dtype, opencv_cuda.dtype);

  cv::Mat opencv_cuda_cpu(mat_opencv_cuda.GetOpenCVCudaMat()->size(), mat_opencv_cuda.GetOpenCVCudaMat()->type());
  mat_opencv_cuda.GetOpenCVCudaMat()->download(opencv_cuda_cpu);

  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(opencv_cuda_cpu.ptr()), opencv.Numel());
}
#endif

#ifdef ENABLE_FLYCV
TEST(fastdeploy, flycv_norm1) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);

  std::vector<float> mean({0.25, 0.35, 0.45});
  std::vector<float> std({0.33, 0.22, 0.54});
  std::vector<float> min;
  std::vector<float> max;
  vision::Normalize::Run(&mat_opencv, mean, std, true, min, max, vision::ProcLib::OPENCV);
  vision::Normalize::Run(&mat_flycv, mean, std, true, min, max, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel());
  check_type(opencv.dtype, flycv.dtype);
}

TEST(fastdeploy, flycv_norm2) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);

  std::vector<float> mean({0.25, 0.35, 0.45});
  std::vector<float> std({0.33, 0.22, 0.54});
  std::vector<float> min({1, 2, 3});
  std::vector<float> max({300, 400, 500});
  vision::Normalize::Run(&mat_opencv, mean, std, true, min, max, vision::ProcLib::OPENCV);
  vision::Normalize::Run(&mat_flycv, mean, std, true, min, max, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel());
  check_type(opencv.dtype, flycv.dtype);
}

TEST(fastdeploy, flycv_norm3) {
  CheckShape check_shape;
  CheckData check_data;
  CheckType check_type;

  cv::Mat mat(64, 64, CV_8UC3);
  cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::Mat mat1 = mat.clone();

  vision::Mat mat_opencv(mat);
  vision::Mat mat_flycv(mat1);

  std::vector<float> mean({0.25, 0.35, 0.45});
  std::vector<float> std({0.33, 0.22, 0.54});
  std::vector<float> min({1, 2, 3});
  std::vector<float> max({300, 400, 500});
  vision::Normalize::Run(&mat_opencv, mean, std, true, min, max, vision::ProcLib::OPENCV);
  vision::Normalize::Run(&mat_flycv, mean, std, true, min, max, vision::ProcLib::FLYCV);

  FDTensor opencv;
  FDTensor flycv;

  mat_opencv.ShareWithTensor(&opencv);
  mat_flycv.ShareWithTensor(&flycv);

  check_shape(opencv.shape, flycv.shape);
  check_data(reinterpret_cast<const float*>(opencv.Data()), reinterpret_cast<const float*>(flycv.Data()), opencv.Numel());
  check_type(opencv.dtype, flycv.dtype);
}
#endif

}  // namespace fastdeploy
