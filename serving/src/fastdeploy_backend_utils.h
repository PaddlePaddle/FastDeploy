
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

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fastdeploy/core/fd_type.h"
#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace triton {
namespace backend {
namespace fastdeploy_runtime {

#define FD_RESPOND_ALL_AND_SET_TRUE_IF_ERROR(                                  \
RESPONSES, RESPONSES_COUNT, BOOL, X)                                           \
  do {                                                                         \
    TRITONSERVER_Error* raasnie_err__ = (X);                                   \
    if (raasnie_err__ != nullptr) {                                            \
      BOOL = true;                                                             \
      for (size_t ridx = 0; ridx < RESPONSES_COUNT; ++ridx) {                  \
        if (RESPONSES[ridx] != nullptr) {                                      \
          LOG_IF_ERROR(                                                        \
              TRITONBACKEND_ResponseSend(RESPONSES[ridx],                      \
                                         TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                                         raasnie_err__),                       \
              "failed to send error response");                                \
          RESPONSES[ridx] = nullptr;                                           \
        }                                                                      \
      }                                                                        \
      TRITONSERVER_ErrorDelete(raasnie_err__);                                 \
    }                                                                          \
  } while (false)

fastdeploy::FDDataType ConvertDataTypeToFD(TRITONSERVER_DataType dtype);

TRITONSERVER_DataType ConvertFDType(fastdeploy::FDDataType dtype);

fastdeploy::FDDataType ModelConfigDataTypeToFDType(
    const std::string& data_type_str);

std::string FDTypeToModelConfigDataType(fastdeploy::FDDataType data_type);

TRITONSERVER_Error* FDParseShape(triton::common::TritonJson::Value& io,
                                 const std::string& name,
                                 std::vector<int32_t>* shape);

}  // namespace fastdeploy_runtime
}  // namespace backend
}  // namespace triton
