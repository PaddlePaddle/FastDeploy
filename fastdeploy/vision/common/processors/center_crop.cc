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

#include "fastdeploy/vision/common/processors/center_crop.h"

#ifdef ENABLE_CVCUDA
#include <cvcuda/OpCustomCrop.hpp>

#include "fastdeploy/vision/common/processors/cvcuda_utils.h"
#endif

namespace fastdeploy {
namespace vision {

bool CenterCrop::ImplByOpenCV(Mat* mat) {
  cv::Mat* im = mat->GetOpenCVMat();
  int height = static_cast<int>(im->rows);
  int width = static_cast<int>(im->cols);
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cv::Rect crop_roi(offset_x, offset_y, width_, height_);
  cv::Mat new_im = (*im)(crop_roi).clone();
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}

#ifdef ENABLE_FLYCV
bool CenterCrop::ImplByFlyCV(Mat* mat) {
  fcv::Mat* im = mat->GetFlyCVMat();
  int height = static_cast<int>(im->height());
  int width = static_cast<int>(im->width());
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }
  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  fcv::Rect crop_roi(offset_x, offset_y, width_, height_);
  fcv::Mat new_im;
  fcv::crop(*im, new_im, crop_roi);
  mat->SetMat(new_im);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  return true;
}
#endif

#ifdef ENABLE_CVCUDA
bool CenterCrop::ImplByCvCuda(Mat* mat) {
  std::cout << Name() << " cvcuda" << std::endl;
  cv::Mat* im = mat->GetOpenCVMat();
  int height = im->rows;
  int width = im->cols;
  int channel = im->channels();
  if (height < height_ || width < width_) {
    FDERROR << "[CenterCrop] Image size less than crop size" << std::endl;
    return false;
  }

  int batchsize = 1;

  // Prepare input tensor
  FDTensor tensor;
  std::string buf_name = Name() + "_cvcuda_src";
  if (mat->device == Device::GPU) {
    mat->ShareWithTensor(&tensor);
  } else if (mat->device == Device::CPU) {
    tensor = *UpdateAndGetReusedBuffer({height, width, channel}, im->type(),
                                       buf_name, Device::GPU);
    FDASSERT(cudaMemcpyAsync(tensor.Data(), im->ptr(), tensor.Nbytes(),
                             cudaMemcpyHostToDevice, mat->Stream()) == 0,
             "[ERROR] Error occurs while copy memory from CPU to GPU.");
  } else {
    FDERROR << "FDMat is on unsupported device: " << mat->device << std::endl;
    return false;
  }
  auto src_tensor = CreateCvCudaTensorWrapData(tensor);

  // Prepare output tensor
  buf_name = Name() + "_cvcuda_dst";
  FDTensor* dst = UpdateAndGetReusedBuffer({height_, width_, channel},
                                           im->type(), buf_name, Device::GPU);
  auto dst_tensor = CreateCvCudaTensorWrapData(*dst);

  int offset_x = static_cast<int>((width - width_) / 2);
  int offset_y = static_cast<int>((height - height_) / 2);
  cvcuda::CustomCrop crop_op;
  NVCVRectI crop_roi = {offset_x, offset_y, width_, height_};
  crop_op(mat->Stream(), src_tensor, dst_tensor, crop_roi);

  cv::Mat out(height_, width_, im->type(), GetCvCudaTensorDataPtr(dst_tensor));

  mat->SetMat(out);
  mat->SetWidth(width_);
  mat->SetHeight(height_);
  mat->device = Device::GPU;
  return true;
}
#endif

bool CenterCrop::Run(Mat* mat, const int& width, const int& height,
                     ProcLib lib) {
  auto c = CenterCrop(width, height);
  return c(mat, lib);
}

}  // namespace vision
}  // namespace fastdeploy
