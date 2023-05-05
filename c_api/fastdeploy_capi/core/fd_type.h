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

#include "fastdeploy_capi/runtime/enum_variables.h"
#include "fastdeploy_capi/core/fd_common.h"

typedef struct FD_C_OneDimArrayUint8 {
  size_t size;
  uint8_t* data;
} FD_C_OneDimArrayUint8;  // std::vector<uint8>

typedef struct FD_C_OneDimArrayInt8 {
  size_t size;
  int8_t* data;
} FD_C_OneDimArrayInt8;  // std::vector<int8>

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

typedef struct FD_C_TwoDimArrayCstr {
  size_t size;
  FD_C_OneDimArrayCstr* data;
} FD_C_TwoDimArrayCstr;  // std::vector<std::vector<std::string>>

typedef struct FD_C_TwoDimArraySize {
  size_t size;
  FD_C_OneDimArraySize* data;
} FD_C_TwoDimArraySize;  // std::vector<std::vector<size_t>>

typedef struct FD_C_TwoDimArrayInt8 {
  size_t size;
  FD_C_OneDimArrayInt8* data;
} FD_C_TwoDimArrayInt8;  // std::vector<std::vector<int8>>

typedef struct FD_C_TwoDimArrayInt32 {
  size_t size;
  FD_C_OneDimArrayInt32* data;
} FD_C_TwoDimArrayInt32;  // std::vector<std::vector<int32_t>>

typedef struct FD_C_ThreeDimArrayInt32 {
  size_t size;
  FD_C_TwoDimArrayInt32* data;
} FD_C_ThreeDimArrayInt32;  // std::vector<std::vector<std::vector<int32_t>>>

typedef struct FD_C_TwoDimArrayFloat {
  size_t size;
  FD_C_OneDimArrayFloat* data;
} FD_C_TwoDimArrayFloat;  // std::vector<std::vector<float>>

typedef void* FD_C_Mat;

typedef struct FD_C_OneDimMat {
  size_t size;
  FD_C_Mat* data;
} FD_C_OneDimMat;

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_DESTROY_FD_TYPE_FUNCTION(typename) FASTDEPLOY_CAPI_EXPORT extern void FD_C_Destroy##typename (__fd_take FD_C_##typename *)
#define DECLARE_AND_IMPLEMENT_FD_TYPE_ONEDIMARRAY(typename) void FD_C_Destroy##typename (__fd_take FD_C_##typename * ptr) \
  { \
     delete[] ptr->data; \
  }

#define DECLARE_AND_IMPLEMENT_FD_TYPE_TWODIMARRAY(typename, one_dim_type) void FD_C_Destroy##typename (__fd_take FD_C_##typename * ptr) \
  { \
     for(int i=0; i< ptr->size; i++) { \
        FD_C_Destroy##one_dim_type(ptr->data + i); \
     } \
     delete[] ptr->data; \
  }

#define DECLARE_AND_IMPLEMENT_FD_TYPE_THREEDIMARRAY(typename, two_dim_type) void FD_C_Destroy##typename (__fd_take FD_C_##typename * ptr) \
  { \
     for(int i=0; i< ptr->size; i++) { \
        FD_C_Destroy##two_dim_type(ptr->data + i); \
     } \
     delete[] ptr->data; \
  }

// FD_C_OneDimArrayUint8
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayUint8);
// FD_C_OneDimArrayInt8
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayInt8);
// FD_C_OneDimArrayInt32
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayInt32);
// FD_C_OneDimArraySize
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArraySize);
// FD_C_OneDimArrayInt64
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayInt64);
// FD_C_OneDimArrayFloat
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayFloat);
// FD_C_Cstr
DECLARE_DESTROY_FD_TYPE_FUNCTION(Cstr);
// FD_C_OneDimArrayCstr
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimArrayCstr);
// FD_C_TwoDimArrayCstr
DECLARE_DESTROY_FD_TYPE_FUNCTION(TwoDimArrayCstr);
// FD_C_TwoDimArraySize
DECLARE_DESTROY_FD_TYPE_FUNCTION(TwoDimArraySize);
// FD_C_TwoDimArrayInt8
DECLARE_DESTROY_FD_TYPE_FUNCTION(TwoDimArrayInt8);
// FD_C_TwoDimArrayInt32
DECLARE_DESTROY_FD_TYPE_FUNCTION(TwoDimArrayInt32);
// FD_C_ThreeDimArrayInt32
DECLARE_DESTROY_FD_TYPE_FUNCTION(ThreeDimArrayInt32);
// FD_C_TwoDimArrayFloat
DECLARE_DESTROY_FD_TYPE_FUNCTION(TwoDimArrayFloat);
// FD_C_OneDimMat
DECLARE_DESTROY_FD_TYPE_FUNCTION(OneDimMat);

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_Mat
FD_C_Imread(const char* imgpath);

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_Imwrite(const char* savepath,
                                                     __fd_keep FD_C_Mat);

FASTDEPLOY_CAPI_EXPORT extern void FD_C_DestroyMat(__fd_take FD_C_Mat mat);

#ifdef __cplusplus
}
#endif
