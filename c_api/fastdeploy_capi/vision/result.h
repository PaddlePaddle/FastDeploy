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

#pragma

#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/fd_type.h"

typedef struct FD_ClassifyResult FD_ClassifyResult;
typedef struct FD_ClassifyResultData FD_ClassifyResultData;

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Create a new FD_ClassifyResult
 *
 * \return Return a pointer to FD_ClassifyResult object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_ClassifyResult* FD_CreateClassifyResult();

/** \brief Destroy a FD_ClassifyResult object
 *
 * \param[in] fd_runtime_option pointer to FD_ClassifyResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyClassifyResult(__fd_take FD_ClassifyResult* fd_classify_result);


/** \brief Get data in FD_ClassifyResult object
 *
 * \param[in] fd_classify_result pointer to FD_ClassifyResult object
 * \param[in] fd_classify_result_data pointer to FD_ClassifyResultData object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_ClassifyResultData* FD_ClassifyResultGetData(__fd_keep FD_ClassifyResult* fd_classify_result);

/** \brief Destroy a FD_ClassifyResultData object
 *
 * \param[in] fd_runtime_option pointer to FD_ClassifyResultData object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyClassifyResultData(__fd_take FD_ClassifyResultData* fd_classify_result_data);


#ifdef __cplusplus
}  // extern "C"
#endif
