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

#include "fastdeploy/vision/common/processors/hwc2chw.h"
#include "fastdeploy/function/transpose.h"

namespace fastdeploy {
namespace vision {
bool HWC2CHW::ImplByOpenCV(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  cv::Mat* im = mat->GetOpenCVMat();
  cv::Mat im_clone = im->clone();
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();

  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(
        im_clone,
        cv::Mat(rh, rw, im->type() % 8,
                im->ptr() + i * rh * rw * FDDataTypeSize(mat->Type())),
        i);
  }
  mat->layout = Layout::CHW;
  return true;
}

#ifdef ENABLE_OPENCV_CUDA
bool HWC2CHW::ImplByOpenCVCuda(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!"
            << std::endl;
    return false;
  }
  cv::cuda::GpuMat* im = mat->GetOpenCVCudaMat();
  auto stream = GetCudaStream();
  std::vector<cv::cuda::GpuMat> split_im;
  cv::cuda::split(*im, split_im, stream);
  int hw_data_size = im->rows * im->cols * FDDataTypeSize(mat->Type());

  for (int c = 0; c < im->channels(); c++) {
    cv::cuda::GpuMat tmp(split_im[c].size(), split_im[c].type(), im->ptr<uint8_t>() + c * hw_data_size);
    split_im[c].copyTo(tmp, stream);
  }
  mat->layout = Layout::CHW;
  return true;
}
#endif

#ifdef ENABLE_FLYCV
bool HWC2CHW::ImplByFalconCV(Mat* mat) {
  if (mat->layout != Layout::HWC) {
    FDERROR << "HWC2CHW: The input data is not Layout::HWC format!" << std::endl;
    return false;
  }
  if (mat->Type() != FDDataType::FP32) {
    FDERROR << "HWC2CHW: Only support float data while use FalconCV, but now it's " << mat->Type() << "." << std::endl;
    return false;
  }
  fcv::Mat* im = mat->GetFalconCVMat();
  fcv::Mat new_im;
  fcv::normalize_to_submean_to_reorder(*im, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, std::vector<uint32_t>(), new_im, false);
  mat->SetMat(new_im);
  mat->layout = Layout::CHW;
  return true;
}
#endif

bool HWC2CHW::Run(Mat* mat, ProcLib lib) {
  auto h = HWC2CHW();
  return h(mat, lib);
}

} // namespace vision
} // namespace fastdeploy
