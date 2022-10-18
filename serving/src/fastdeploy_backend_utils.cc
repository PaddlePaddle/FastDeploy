// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "fastdeploy_backend_utils.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>

namespace triton {
namespace backend {
namespace fastdeploy_runtime {

TRITONSERVER_DataType ConvertFDType(fastdeploy::FDDataType dtype) {
  switch (dtype) {
    case fastdeploy::FDDataType::UNKNOWN1:
      return TRITONSERVER_TYPE_INVALID;
    case ::fastdeploy::FDDataType::UINT8:
      return TRITONSERVER_TYPE_UINT8;
    case ::fastdeploy::FDDataType::INT8:
      return TRITONSERVER_TYPE_INT8;
    case ::fastdeploy::FDDataType::INT32:
      return TRITONSERVER_TYPE_INT32;
    case ::fastdeploy::FDDataType::INT64:
      return TRITONSERVER_TYPE_INT64;
    case ::fastdeploy::FDDataType::FP32:
      return TRITONSERVER_TYPE_FP32;
    case ::fastdeploy::FDDataType::FP16:
      return TRITONSERVER_TYPE_FP16;
    default:
      break;
  }
  return TRITONSERVER_TYPE_INVALID;
}

fastdeploy::FDDataType ConvertDataTypeToFD(TRITONSERVER_DataType dtype) {
  switch (dtype) {
    case TRITONSERVER_TYPE_INVALID:
      return ::fastdeploy::FDDataType::UNKNOWN1;
    case TRITONSERVER_TYPE_UINT8:
      return ::fastdeploy::FDDataType::UINT8;
    case TRITONSERVER_TYPE_INT8:
      return ::fastdeploy::FDDataType::INT8;
    case TRITONSERVER_TYPE_INT32:
      return ::fastdeploy::FDDataType::INT32;
    case TRITONSERVER_TYPE_INT64:
      return ::fastdeploy::FDDataType::INT64;
    case TRITONSERVER_TYPE_FP32:
      return ::fastdeploy::FDDataType::FP32;
    case TRITONSERVER_TYPE_FP16:
      return ::fastdeploy::FDDataType::FP16;
    default:
      break;
  }
  return ::fastdeploy::FDDataType::UNKNOWN1;
}

fastdeploy::FDDataType ModelConfigDataTypeToFDType(
    const std::string& data_type_str) {
  // Must start with "TYPE_".
  if (data_type_str.rfind("TYPE_", 0) != 0) {
    return fastdeploy::FDDataType::UNKNOWN1;
  }

  const std::string dtype = data_type_str.substr(strlen("TYPE_"));

  if (dtype == "UINT8") {
    return fastdeploy::FDDataType::UINT8;
  } else if (dtype == "INT8") {
    return fastdeploy::FDDataType::INT8;
  } else if (dtype == "INT32") {
    return fastdeploy::FDDataType::INT32;
  } else if (dtype == "INT64") {
    return fastdeploy::FDDataType::INT64;
  } else if (dtype == "FP16") {
    return fastdeploy::FDDataType::FP16;
  } else if (dtype == "FP32") {
    return fastdeploy::FDDataType::FP32;
  }
  return fastdeploy::FDDataType::UNKNOWN1;
}

std::string FDTypeToModelConfigDataType(fastdeploy::FDDataType data_type) {
  if (data_type == fastdeploy::FDDataType::UINT8) {
    return "TYPE_UINT8";
  } else if (data_type == fastdeploy::FDDataType::INT8) {
    return "TYPE_INT8";
  } else if (data_type == fastdeploy::FDDataType::INT32) {
    return "TYPE_INT32";
  } else if (data_type == fastdeploy::FDDataType::INT64) {
    return "TYPE_INT64";
  } else if (data_type == fastdeploy::FDDataType::FP16) {
    return "TYPE_FP16";
  } else if (data_type == fastdeploy::FDDataType::FP32) {
    return "TYPE_FP32";
  }

  return "TYPE_INVALID";
}

TRITONSERVER_Error* FDParseShape(triton::common::TritonJson::Value& io,
                                 const std::string& name,
                                 std::vector<int32_t>* shape) {
  std::string shape_string;
  RETURN_IF_ERROR(io.MemberAsString(name.c_str(), &shape_string));

  std::vector<std::string> str_shapes;
  std::istringstream in(shape_string);
  std::copy(std::istream_iterator<std::string>(in),
            std::istream_iterator<std::string>(),
            std::back_inserter(str_shapes));

  std::transform(str_shapes.cbegin(), str_shapes.cend(),
                 std::back_inserter(*shape),
                 [](const std::string& str) -> int32_t {
                   return static_cast<int32_t>(std::stoll(str));
                 });

  return nullptr;  // success
}

}  // namespace fastdeploy_runtime
}  // namespace backend
}  // namespace triton