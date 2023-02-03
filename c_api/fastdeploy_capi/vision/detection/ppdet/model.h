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

#include "fastdeploy_capi/fd_type.h"
#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/vision/result.h"
#include "fastdeploy_capi/runtime_option.h"

typedef struct FD_PPYOLOE FD_PPYOLOE;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a new FD_PPYOLOE
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_PPYOLOE object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_PPYOLOE* FD_CreatesPPYOLOE(const char* model_file, const char* params_file,
                                                                                    const char* config_file,
                                                                                    FD_RuntimeOption* fd_runtime_option,
                                                                                    const FD_ModelFormat model_format);

/** \brief Destroy a FD_PPYOLOE object
 *
 * \param[in] fd_ppyoloe pointer to FD_PPYOLOE object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyPPYOLOE(__fd_take FD_PPYOLOE* fd_ppyoloe);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_ppyoloe pointer to FD_PPYOLOE object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_PPYOLOEPredict(__fd_take FD_PPYOLOE* fd_ppyoloe,
                                    FD_Mat* img, FD_DetectionResult* fd_classify_result);

#ifdef __cplusplus
}  // extern "C"
#endif
