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

#define DECLARE_CREATE_WRAPPER_FUNCTION(model_type) FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_##model_type##Wrapper* \
FD_C_Create##model_type##Wrapper( \
    const char* model_file, const char* params_file, const char* config_file, \
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper, \
    const FD_C_ModelFormat model_format)


#define DECLARE_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) FASTDEPLOY_CAPI_EXPORT extern void \
FD_C_Destroy##model_type##Wrapper(__fd_take FD_C_##model_type##Wrapper* wrapper_var_name);

#define DECLARE_PREDICT_FUNCTION(model_type, wrapper_var_name) FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_##model_type##WrapperPredict( \
    __fd_take FD_C_##model_type##Wrapper* wrapper_var_name, FD_C_Mat img, \
    FD_C_DetectionResult* fd_c_detection_result)

#define DECLARE_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    __fd_keep FD_C_##model_type##Wrapper* wrapper_var_name)


#define DECLARE_BATCH_PREDICT_FUNCTION(model_type, wrapper_var_name) FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_##model_type##WrapperBatchPredict( \
                            __fd_keep FD_C_##model_type##Wrapper* wrapper_var_name, \
                            FD_C_OneDimMat imgs, \
                            FD_C_OneDimDetectionResult* results)


#define DECLARE_AND_IMPLEMENT_CREATE_WRAPPER_FUNCTION(model_type, var_name) FD_C_##model_type##Wrapper* FD_C_Create##model_type##Wrapper(\
    const char* model_file, const char* params_file, const char* config_file, \
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper, \
    const FD_C_ModelFormat model_format) { \
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper, \
                                                   fd_c_runtime_option_wrapper); \
  FD_C_##model_type##Wrapper* fd_c_##model_type##_wrapper = new FD_C_##model_type##Wrapper(); \
  fd_c_##model_type##_wrapper->var_name = \
      std::unique_ptr<fastdeploy::vision::detection::model_type>( \
          new fastdeploy::vision::detection::model_type( \
              std::string(model_file), std::string(params_file), \
              std::string(config_file), *runtime_option, \
              static_cast<fastdeploy::ModelFormat>(model_format))); \
  return fd_c_##model_type##_wrapper;\
}

#define DECLARE_AND_IMPLEMENT_DESTROY_WRAPPER_FUNCTION(model_type, wrapper_var_name) void FD_C_Destroy##model_type##Wrapper( \
    __fd_take FD_C_##model_type##Wrapper* wrapper_var_name) { \
  delete wrapper_var_name; \
}


#define DECLARE_AND_IMPLEMENT_PREDICT_FUNCTION(model_type, wrapper_var_name) FD_C_Bool FD_C_##model_type##WrapperPredict( \
    FD_C_##model_type##Wrapper* wrapper_var_name, FD_C_Mat img, \
    FD_C_DetectionResult* fd_c_detection_result) { \
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);                               \
  auto& model =                                                                \
      CHECK_AND_CONVERT_FD_TYPE(model_type##Wrapper, wrapper_var_name);        \
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =                 \
      FD_C_CreateDetectionResultWrapper();                                     \
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(                          \
      DetectionResultWrapper, fd_c_detection_result_wrapper);                  \
  bool successful = model->Predict(im, detection_result.get());                \
  if (successful) {                                                            \
        FD_C_DetectionResultWrapperToCResult(fd_c_detection_result_wrapper, fd_c_detection_result); \
  } \
FD_C_DestroyDetectionResultWrapper(fd_c_detection_result_wrapper); \
return successful; \
}

#define DECLARE_AND_IMPLEMENT_INITIALIZED_FUNCTION(model_type, wrapper_var_name)  FD_C_Bool FD_C_##model_type##WrapperInitialized( \
    FD_C_##model_type##Wrapper* wrapper_var_name) { \
    auto& model = \
      CHECK_AND_CONVERT_FD_TYPE(model_type##Wrapper, wrapper_var_name); \
    return model->Initialized(); \
}

#define DECLARE_AND_IMPLEMENT_BATCH_PREDICT_FUNCTION(model_type, wrapper_var_name) FD_C_Bool FD_C_##model_type##WrapperBatchPredict( \
    FD_C_##model_type##Wrapper* wrapper_var_name, FD_C_OneDimMat imgs, \
    FD_C_OneDimDetectionResult* results) { \
  std::vector<cv::Mat> imgs_vec; \
  std::vector<fastdeploy::vision::DetectionResult> results_out; \
  std::vector<FD_C_DetectionResultWrapper*> results_wrapper_out; \
  for (int i = 0; i < imgs.size; i++) { \
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i]))); \
    FD_C_DetectionResultWrapper* fd_detection_result_wrapper = FD_C_CreateDetectionResultWrapper(); \
    results_wrapper_out.push_back(fd_detection_result_wrapper); \
  } \
  auto& model = \
      CHECK_AND_CONVERT_FD_TYPE(model_type##Wrapper, wrapper_var_name); \
  bool successful = model->BatchPredict(imgs_vec, &results_out); \
  if (successful) { \
    results->size = results_out.size(); \
    results->data = new FD_C_DetectionResult[results->size]; \
    for (int i = 0; i < results_out.size(); i++) { \
      (*CHECK_AND_CONVERT_FD_TYPE(DetectionResultWrapper, \
                                  results_wrapper_out[i])) = std::move(results_out[i]); \
      FD_C_DetectionResultWrapperToCResult(results_wrapper_out[i], &results->data[i]); \
    } \
  } \
  for (int i = 0; i < results_out.size(); i++) { \
    FD_C_DestroyDetectionResultWrapper(results_wrapper_out[i]); \
  }\
  return successful; \
}
