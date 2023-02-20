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

#define OCR_DECLARE_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) FASTDEPLOY_CAPI_EXPORT extern void \
FD_C_Destroy##model_type##Wrapper(__fd_take FD_C_##model_type##Wrapper* wrapper_var_name);

#define OCR_DECLARE_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    __fd_keep FD_C_##model_type##Wrapper* wrapper_var_name)

#define OCR_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) delete wrapper_var_name

#define OCR_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) void FD_C_Destroy##model_type##Wrapper( \
    __fd_take FD_C_##model_type##Wrapper* wrapper_var_name) { \
  OCR_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name); \
}

#define OCR_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name)   auto& model = \
      CHECK_AND_CONVERT_FD_TYPE(model_type##Wrapper, wrapper_var_name); \
return model->Initialized();

#define OCR_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    FD_C_##model_type##Wrapper* wrapper_var_name) { \
  OCR_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name); \
}

#define PIPELINE_DECLARE_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) FASTDEPLOY_CAPI_EXPORT extern void \
FD_C_Destroy##model_type##Wrapper(__fd_take FD_C_##model_type##Wrapper* wrapper_var_name);

#define PIPELINE_DECLARE_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    __fd_keep FD_C_##model_type##Wrapper* wrapper_var_name)

#define PIPELINE_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) delete wrapper_var_name

#define PIPELINE_DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) void FD_C_Destroy##model_type##Wrapper( \
    __fd_take FD_C_##model_type##Wrapper* wrapper_var_name) { \
  PIPELINE_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name); \
}

#define PIPELINE_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name)   auto& model = \
      CHECK_AND_CONVERT_FD_TYPE(model_type##Wrapper, wrapper_var_name); \
return model->Initialized();

#define PIPELINE_DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    FD_C_##model_type##Wrapper* wrapper_var_name) { \
  PIPELINE_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name); \
}
