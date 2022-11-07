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

#include "fastdeploy/vision/common/processors/normalize.h"

namespace fastdeploy {
namespace vision {
Normalize::Normalize(const std::vector<float>& mean,
                     const std::vector<float>& std, bool is_scale,
                     const std::vector<float>& min,
                     const std::vector<float>& max) {
  FDASSERT(mean.size() == std.size(),
           "Normalize: requires the size of mean equal to the size of std.");
  std::vector<double> mean_(mean.begin(), mean.end());
  std::vector<double> std_(std.begin(), std.end());
  std::vector<double> min_(mean.size(), 0.0);
  std::vector<double> max_(mean.size(), 255.0);
  if (min.size() != 0) {
    FDASSERT(
        min.size() == mean.size(),
        "Normalize: while min is defined, requires the size of min equal to "
        "the size of mean.");
    min_.assign(min.begin(), min.end());
  }
  if (max.size() != 0) {
    FDASSERT(
        min.size() == mean.size(),
        "Normalize: while max is defined, requires the size of max equal to "
        "the size of mean.");
    max_.assign(max.begin(), max.end());
  }
  for (auto c = 0; c < mean_.size(); ++c) {
    double alpha = 1.0;
    if (is_scale) {
      alpha /= (max_[c] - min_[c]);
    }
    double beta = -1.0 * (mean_[c] + min_[c] * alpha) / std_[c];
    alpha /= std_[c];
    alpha_.push_back(alpha);
    beta_.push_back(beta);
  }
}

bool Normalize::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();

  std::vector<cv::Mat> split_im;
  cv::split(*im, split_im);
  for (int c = 0; c < im->channels(); c++) {
    split_im[c].convertTo(split_im[c], CV_32FC1, alpha_[c], beta_[c]);
  }
  cv::merge(split_im, *im);
  return true;
}

#ifdef ENABLE_FLYCV
bool Normalize::ImplByFlyCV(Mat* mat) {
  fcv::Mat* im = mat->GetFlyCVMat();
  if (im->channels() != 3) {
    FDERROR << "Only supports 3-channels image in FlyCV, but now it's "
            << im->channels() << "." << std::endl;
    return false;
  }

  std::vector<float> mean(3, 0);
  std::vector<float> std(3, 0);
  for (size_t i = 0; i < 3; ++i) {
    std[i] = 1.0 / alpha_[i];
    mean[i] = -1 * beta_[i] * std[i];
  }
  fcv::Mat new_im(im->width(), im->height(),
                  fcv::FCVImageType::PKG_BGR_F32);
  fcv::normalize_to_submean_to_reorder(*im, mean, std, std::vector<uint32_t>(),
                                       new_im, true);
  mat->SetMat(new_im);
  return true;
}
#endif

bool Normalize::Run(Mat* mat, const std::vector<float>& mean,
                    const std::vector<float>& std, bool is_scale,
                    const std::vector<float>& min,
                    const std::vector<float>& max, ProcLib lib) {
  auto n = Normalize(mean, std, is_scale, min, max);
  return n(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
