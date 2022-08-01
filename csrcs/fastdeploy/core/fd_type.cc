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

#include "fastdeploy/core/fd_type.h"
#include "fastdeploy/utils/utils.h"

namespace fastdeploy {

int FDDataTypeSize(const FDDataType& data_type) {
  FDASSERT(data_type != FDDataType::FP16, "Float16 is not supported.");
  if (data_type == FDDataType::BOOL) {
    return sizeof(bool);
  } else if (data_type == FDDataType::INT16) {
    return sizeof(int16_t);
  } else if (data_type == FDDataType::INT32) {
    return sizeof(int32_t);
  } else if (data_type == FDDataType::INT64) {
    return sizeof(int64_t);
  } else if (data_type == FDDataType::FP32) {
    return sizeof(float);
  } else if (data_type == FDDataType::FP64) {
    return sizeof(double);
  } else if (data_type == FDDataType::UINT8) {
    return sizeof(uint8_t);
  } else {
    FDASSERT(false, "Unexpected data type: " + Str(data_type));
  }
  return -1;
}

std::string Str(const Device& d) {
  std::string out;
  switch (d) {
    case Device::DEFAULT:
      out = "Device::DEFAULT";
      break;
    case Device::CPU:
      out = "Device::CPU";
      break;
    case Device::GPU:
      out = "Device::GPU";
      break;
    default:
      out = "Device::UNKOWN";
  }
  return out;
}

std::string Str(const FDDataType& fdt) {
  std::string out;
  switch (fdt) {
    case FDDataType::BOOL:
      out = "FDDataType::BOOL";
      break;
    case FDDataType::INT16:
      out = "FDDataType::INT16";
      break;
    case FDDataType::INT32:
      out = "FDDataType::INT32";
      break;
    case FDDataType::INT64:
      out = "FDDataType::INT64";
      break;
    case FDDataType::FP32:
      out = "FDDataType::FP32";
      break;
    case FDDataType::FP64:
      out = "FDDataType::FP64";
      break;
    case FDDataType::FP16:
      out = "FDDataType::FP16";
      break;
    case FDDataType::UINT8:
      out = "FDDataType::UINT8";
      break;
    case FDDataType::INT8:
      out = "FDDataType::INT8";
      break;
    default:
      out = "FDDataType::UNKNOWN";
  }
  return out;
}

}  // namespace fastdeploy
