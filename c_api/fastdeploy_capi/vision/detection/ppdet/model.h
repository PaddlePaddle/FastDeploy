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

#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/fd_type.h"
#include "fastdeploy_capi/runtime_option.h"
#include "fastdeploy_capi/vision/result.h"

typedef struct FD_C_PPYOLOEWrapper FD_C_PPYOLOEWrapper;
typedef struct FD_C_RuntimeOptionWrapper FD_C_RuntimeOptionWrapper;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a new FD_C_PPYOLOEWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PPYOLOEWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_C_PPYOLOEWrapper*
FD_C_CreatesPPYOLOEWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format);

/** \brief Destroy a FD_C_PPYOLOEWrapper object
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern void
FD_C_DestroyPPYOLOEWrapper(__fd_take FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

FASTDEPLOY_CAPI_EXPORT extern FD_C_Bool FD_C_PPYOLOEWrapperPredict(
    __fd_take FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper, FD_C_Mat img,
    FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper);

#ifdef __cplusplus
}  // extern "C"
#endif
