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

#pragma once

#include <stdint.h>
#include <stdio.h>

#include "fastdeploy_capi/fd_common.h"  // NOLINT

typedef struct FD_C_OneDimArrayUint8 {
  size_t size;
  uint8_t* data;
} FD_C_OneDimArrayUint8;  // std::vector<int32_t>

typedef struct FD_C_OneDimArrayInt32 {
  size_t size;
  int32_t* data;
} FD_C_OneDimArrayInt32;  // std::vector<int32_t>

typedef struct FD_C_OneDimArraySize {
  size_t size;
  size_t* data;
} FD_C_OneDimArraySize;  // std::vector<size_t>

typedef struct FD_C_OneDimArrayInt64 {
  size_t size;
  int64_t* data;
} FD_C_OneDimArrayInt64;  // std::vector<int64_t>

typedef struct FD_C_OneDimArrayFloat {
  size_t size;
  float* data;
} FD_C_OneDimArrayFloat;  // std::vector<float>

typedef struct FD_C_Cstr {
  size_t size;
  char* data;
} FD_C_Cstr;  // std::string

typedef struct FD_C_OneDimArrayCstr {
  size_t size;
  FD_C_Cstr* data;
} FD_C_OneDimArrayCstr;  // std::vector<std::string>

typedef struct FD_C_TwoDimArraySize {
  size_t size;
  FD_C_OneDimArraySize* data;
} FD_C_TwoDimArraySize;  // std::vector<std::vector<size_t>>

typedef struct FD_C_TwoDimArrayFloat {
  size_t size;
  FD_C_OneDimArrayFloat* data;
} FD_C_TwoDimArrayFloat;  // std::vector<std::vector<float>>

typedef void* FD_C_Mat;
