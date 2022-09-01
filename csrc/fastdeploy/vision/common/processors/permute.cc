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

#include "fastdeploy/vision/common/processors/permute.h"

namespace fastdeploy {
namespace vision {
bool Permute::CpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetCpuMat();
  (*im).convertTo(*im, CV_32FC3);
  cv::Mat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();

  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(
        im_clone,
        cv::Mat(rh, rw, CV_32FC1,
                im->ptr() + i * rh * rw * FDDataTypeSize(mat->Type())),
        i);
  }
  mat->layout = Layout::CHW;
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Permute::GpuRun(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  (*im).convertTo(*im, CV_32FC3);
  cv::cuda::GpuMat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  int num_pixels = rh * rw;
  std::vector<cv::cuda::GpuMat> channels{
      cv::cuda::GpuMat(rh, rw, CV_32FC1, &(im->ptr()[0])),
      cv::cuda::GpuMat(rh, rw, CV_32FC1, &(im->ptr()[num_pixels])),
      cv::cuda::GpuMat(rh, rw, CV_32FC1, &(im->ptr()[num_pixels * 2]))};
  cv::cuda::split(im_clone, channels);
  mat->layout = Layout::CHW;
  return true;
}
#endif

bool Permute::Run(Mat* mat, ProcLib lib) {
  auto h = Permute();
  return h(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
