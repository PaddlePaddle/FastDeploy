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

#include "fastdeploy/vision/common/processors/convert.h"

namespace fastdeploy {

namespace vision {

Convert::Convert(const std::vector<float>& alpha,
                 const std::vector<float>& beta) {
  FDASSERT(alpha.size() == beta.size(),
           "Convert: requires the size of alpha equal to the size of beta.");
  FDASSERT(alpha.size() != 0,
           "Convert: requires the size of alpha and beta > 0.");
  alpha_.assign(alpha.begin(), alpha.end());
  beta_.assign(beta.begin(), beta.end());
}

bool Convert::CpuRun(Mat* mat) {
  cv::Mat* im = mat->GetCpuMat();
  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::Mat new_im;
  cv::merge(split_im, new_im);
  mat->SetMat(new_im);
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool Convert::GpuRun(Mat* mat) {
  cv::cuda::GpuMat* im = mat->GetGpuMat();
  std::vector<cv::cuda::GpuMat> split_im;
  cv::cuda::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::cuda::merge(split_im, *im);
  return true;
}
#endif

bool Convert::Run(Mat* mat, const std::vector<float>& alpha,
                  const std::vector<float>& beta, ProcLib lib) {
  auto c = Convert(alpha, beta);
  return c(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
