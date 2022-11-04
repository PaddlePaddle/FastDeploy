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

#include "fastdeploy/utils/utils.h"

#include "fastdeploy/vision/common/processors/utils.h"

namespace fastdeploy {
namespace vision {

FDDataType OpenCVDataTypeToFD(int type) {
  type = type % 8;
  if (type == 0) {
    return FDDataType::UINT8;
  } else if (type == 1) {
    return FDDataType::INT8;
  } else if (type == 2) {
    FDASSERT(false,
             "While calling OpenCVDataTypeToFD(), get UINT16 type which is not "
             "supported now.");
  } else if (type == 3) {
    return FDDataType::INT16;
  } else if (type == 4) {
    return FDDataType::INT32;
  } else if (type == 5) {
    return FDDataType::FP32;
  } else if (type == 6) {
    return FDDataType::FP64;
  } else {
    FDASSERT(false,
             "While calling OpenCVDataTypeToFD(), get type = %d, which is not "
             "expected.",
             type);
  }
}

#ifdef ENABLE_FLYCV
FDDataType FlyCVDataTypeToFD(fcv::FCVImageType type) {
  if (type == fcv::FCVImageType::GRAY_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLANAR_BGR_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLANAR_RGB_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLANAR_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLANAR_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PLANAR_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLANAR_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLANAR_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PLANAR_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_BGRA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_RGBA_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_BGR565_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::PACKAGE_RGB565_U8) {
    return FDDataType::UINT8;
  } else if (type == fcv::FCVImageType::GRAY_S32) {
    return FDDataType::INT32;
  } else if (type == fcv::FCVImageType::GRAY_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_BGR_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_RGB_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_BGRA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::PACKAGE_RGBA_F32) {
    return FDDataType::FP32;
  } else if (type == fcv::FCVImageType::GRAY_F64) {
    return FDDataType::FP64;
  }
  FDASSERT(false, "While calling FlyCVDataTypeToFD(), get unexpected type:%d.",
           int(type));
  return FDDataType::UNKNOWN1;
}

fcv::FCVImageType CreateFlyCVDataType(FDDataType type, int channel) {
  FDASSERT(channel == 1 || channel == 3 || channel == 4,
           "Only support channel be 1/3/4 in FlyCV.");
  if (type == FDDataType::UINT8) {
    if (channel == 1) {
      return fcv::FCVImageType::GRAY_U8;
    } else if (channel == 3) {
      return fcv::FCVImageType::PACKAGE_BGR_U8;
    } else {
      return fcv::FCVImageType::PACKAGE_BGRA_U8;
    }
  } else if (type == FDDataType::FP32) {
    if (channel == 1) {
      return fcv::FCVImageType::GRAY_F32;
    } else if (channel == 3) {
      return fcv::FCVImageType::PACKAGE_BGR_F32;
    } else {
      return fcv::FCVImageType::PACKAGE_BGRA_F32;
    }
  }
  FDASSERT(false, "Data type of %s is not supported.", Str(type).c_str());
  return fcv::FCVImageType::PACKAGE_BGR_F32;
}

int CreateOpenCVDataType(FDDataType type, int channel) {
  FDASSERT(channel == 1 || channel == 3 || channel == 4,
           "Only support channel be 1/3/4 in OpenCV.");
  if (type == FDDataType::UINT8) {
    if (channel == 1) {
      return CV_8UC1;
    } else if (channel == 3) {
      return CV_8UC3;
    } else {
      return CV_8UC4;
    }
  } else if (type == FDDataType::FP32) {
    if (channel == 1) {
      return CV_32FC1;
    } else if (channel == 3) {
      return CV_32FC3;
    } else {
      return CV_32FC4;
    }
  }
  FDASSERT(false, "Data type of %s is not supported.", Str(type).c_str());
  return CV_32FC3;
}

fcv::Mat ConvertOpenCVMatToFlyCV(cv::Mat& im) {
  int type = im.type() % 8;
  // 0: uint8; 5: float32; 6: float64
  if (type != 0 && type != 5 && type != 6) {
    FDASSERT(false, "Only support type of uint8/float/double, but now it's %d.",
             im.type());
  }
  auto fcv_type =
      CreateFlyCVDataType(OpenCVDataTypeToFD(im.type()), im.channels());
  return fcv::Mat(im.cols, im.rows, fcv_type, im.ptr());  // reference only
}

cv::Mat ConvertFlyCVMatToOpenCV(fcv::Mat& fim) {
  auto fd_dtype = FlyCVDataTypeToFD(fim.type());
  if (fd_dtype != FDDataType::UINT8 && fd_dtype != FDDataType::FP32 &&
      fd_dtype != FDDataType::FP64) {
    FDASSERT(false, "Only support type of uint8/float/double, but now it's %s.",
             Str(fd_dtype).c_str());
  }
  auto ocv_type = CreateOpenCVDataType(fd_dtype, fim.channels());
  return cv::Mat(fim.height(), fim.width(), ocv_type,
                 fim.data());  // reference only
}
#endif

}  // namespace vision
}  // namespace fastdeploy
