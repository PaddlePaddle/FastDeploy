// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy_capi/core/fd_type.h"

#include <opencv2/imgcodecs.hpp>

#include "fastdeploy_capi/core/fd_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// FD_C_OneDimArrayUint8
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArrayUint8)
// FD_C_OneDimArrayInt8
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArrayInt8)
// FD_C_OneDimArrayInt32
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArrayInt32)
// FD_C_OneDimArraySize
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArraySize)
// FD_C_OneDimArrayInt64
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArrayInt64)
// FD_C_OneDimArrayFloat
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(OneDimArrayFloat)
// FD_C_Cstr
DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(Cstr)
// FD_C_OneDimArrayCstr
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(OneDimArrayCstr, Cstr)
// FD_C_TwoDimArraySize
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(TwoDimArraySize, OneDimArraySize)
// FD_C_TwoDimArrayInt8
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(TwoDimArrayInt8, OneDimArrayInt8)
// FD_C_TwoDimArrayInt32
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(TwoDimArrayInt32, OneDimArrayInt32)
// FD_C_ThreeDimArrayInt32
DECLARE_AND_IMPLEMENT_FD_TYPE_THREEDIMARRAY(ThreeDimArrayInt32,
                                            TwoDimArrayInt32)
// FD_C_TwoDimArrayFloat
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(TwoDimArrayFloat, OneDimArrayFloat)
// FD_C_OneDimMat
DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(OneDimMat, Mat)

FD_C_Mat FD_C_Imread(const char* imgpath) {
  cv::Mat image = cv::imread(imgpath);
  return new cv::Mat(image);
}

FD_C_Bool FD_C_Imwrite(const char* savepath, FD_C_Mat img) {
  cv::Mat cv_img = *(reinterpret_cast<cv::Mat*>(img));
  bool result = cv::imwrite(savepath, cv_img);
  return result;
}

void FD_C_DestroyMat(FD_C_Mat mat) { delete reinterpret_cast<cv::Mat*>(mat); }

#ifdef __cplusplus
}
#endif
