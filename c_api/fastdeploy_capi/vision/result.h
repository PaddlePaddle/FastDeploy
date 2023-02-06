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

typedef struct FD_ClassifyResultWrapper FD_ClassifyResultWrapper;
typedef struct FD_DetectionResultWrapper FD_DetectionResultWrapper;

#ifdef __cplusplus
extern "C" {
#endif

PD_ENUM(FD_ResultType) {
  UNKNOWN_RESULT,
  CLASSIFY,
  DETECTION,
  SEGMENTATION,
  OCR,
  MOT,
  FACE_DETECTION,
  FACE_ALIGNMENT,
  FACE_RECOGNITION,
  MATTING,
  MASK,
  KEYPOINT_DETECTION,
  HEADPOSE,
};

typedef struct FD_ClassifyResult{
  FD_OneDimArrayInt32 label_ids;
  FD_OneDimArrayFloat scores;
  FD_ResultType type;
} FD_ClassifyResult;

typedef struct FD_Mask{
  FD_OneDimArrayUint8 data;
  FD_OneDimArrayInt64 shape;
  FD_ResultType type;
} FD_Mask;

typedef struct FD_OneDimMask {
  size_t size;
  FD_Mask* data;
} FD_OneDimMask;  // std::vector<FD_Mask>

typedef struct FD_DetectionResult{
  FD_TwoDimArrayFloat  boxes;
  FD_OneDimArrayFloat scores;
  FD_OneDimArrayInt32 label_ids;
  FD_OneDimMask masks;
  FD_Bool contain_masks;
  FD_ResultType type;
} FD_DetectionResult;

// Classification Results

/** \brief Create a new FD_ClassifyResultWrapper object
 *
 * \return Return a pointer to FD_ClassifyResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_ClassifyResultWrapper* FD_CreateClassifyResultWrapper();

/** \brief Destroy a FD_ClassifyResultWrapper object
 *
 * \param[in] fd_classify_result_wrapper pointer to FD_ClassifyResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyClassifyResultWrapper(__fd_take FD_ClassifyResultWrapper* fd_classify_result_wrapper);

/** \brief Destroy a FD_ClassifyResult object
 *
 * \param[in] fd_classify_result pointer to FD_ClassifyResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyClassifyResult(__fd_take FD_ClassifyResult* fd_classify_result);


/** \brief Get a FD_ClassifyResult object from FD_ClassifyResultWrapper object
 *
 * \param[in] fd_classify_result_wrapper pointer to FD_ClassifyResultWrapper object
 * \return Return a pointer to FD_ClassifyResult object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_ClassifyResult* FD_ClassifyResultWrapperGetData(__fd_keep FD_ClassifyResultWrapper* fd_classify_result_wrapper);

/** \brief Create a new FD_ClassifyResultWrapper object from FD_ClassifyResult object
 *
 * \param[in] fd_classify_result pointer to FD_ClassifyResult object
 * \return Return a pointer to FD_ClassifyResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_ClassifyResultWrapper* FD_CreateClassifyResultWrapperFromData(__fd_keep FD_ClassifyResult* fd_classify_result);

// Detection Results

/** \brief Create a new FD_DetectionResultWrapper object
 *
 * \return Return a pointer to FD_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_DetectionResultWrapper* FD_CreateDetectionResultWrapper();

/** \brief Destroy a FD_DetectionResultWrapper object
 *
 * \param[in] fd_detection_result_wrapper pointer to FD_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyDetectionResultWrapper(__fd_take FD_DetectionResultWrapper* fd_detection_result_wrapper);

/** \brief Destroy a FD_DetectionResult object
 *
 * \param[in] fd_detection_result pointer to FD_DetectionResult object
 */

FASTDEPLOY_CAPI_EXPORT extern void FD_DestroyDetectionResult(__fd_take FD_DetectionResult* fd_detection_result);

/** \brief Get a FD_DetectionResult object from FD_DetectionResultWrapper object
 *
 * \param[in] fd_detection_result_wrapper pointer to FD_DetectionResultWrapper object
 * \return Return a pointer to FD_DetectionResult object
 */
FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_DetectionResult* FD_DetectionResultWrapperGetData(__fd_keep FD_DetectionResultWrapper* fd_detection_result_wrapper);

/** \brief Create a new FD_DetectionResultWrapper object from FD_DetectionResult object
 *
 * \param[in] fd_detection_result pointer to FD_DetectionResult object
 * \return Return a pointer to FD_DetectionResultWrapper object
 */

FASTDEPLOY_CAPI_EXPORT extern __fd_give FD_DetectionResultWrapper* FD_CreateDetectionResultWrapperFromData(__fd_keep FD_DetectionResult* fd_detection_result);




#ifdef __cplusplus
}  // extern "C"
#endif
