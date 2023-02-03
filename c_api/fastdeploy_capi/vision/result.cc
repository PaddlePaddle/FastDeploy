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

#include "fastdeploy_capi/vision/result.h"

#include "fastdeploy/utils/utils.h"
#include "fastdeploy_capi/types_internal.h"

namespace fastdeploy {
std::unique_ptr<fastdeploy::vision::ClassifyResult>&
CheckAndConvertFD_ClassifyResult(FD_ClassifyResult* fd_classify_result) {
  FDASSERT(fd_classify_result != nullptr,
           "The pointer of fd_classify_result shouldn't be nullptr.");
  return fd_classify_result->classify_result;
}

std::unique_ptr<fastdeploy::vision::DetectionResult>&
CheckAndConvertFD_DetectionResult(FD_DetectionResult* fd_detection_result) {
  FDASSERT(fd_detection_result != nullptr,
           "The pointer of fd_detection_result shouldn't be nullptr.");
  return fd_detection_result->detection_result;
}
}  // namespace fastdeploy

extern "C" {

FD_ClassifyResult* FD_CreateClassifyResult() {
  FD_ClassifyResult* fd_classify_result = new FD_ClassifyResult();
  fd_classify_result->classify_result =
      std::unique_ptr<fastdeploy::vision::ClassifyResult>(
          new fastdeploy::vision::ClassifyResult());
  return fd_classify_result;
}

void FD_DestroyClassifyResult(__fd_take FD_ClassifyResult* fd_classify_result) {
  delete fd_classify_result;
}

FD_ClassifyResultData* FD_ClassifyResultGetData(
    __fd_keep FD_ClassifyResult* fd_classify_result) {
  CHECK_AND_CONVERT_FD_ClassifyResult;
  FD_ClassifyResultData* fd_classify_result_data = new FD_ClassifyResultData();
  fd_classify_result_data->label_ids.size = classify_result->label_ids.size();
  // note: FD_ClassifyResultData share the underlying data with
  // FD_ClassifyResult if FD_ClassifyResult is released, fd_classify_result_data
  // should be released too
  fd_classify_result_data->label_ids.data = classify_result->label_ids.data();
  fd_classify_result_data->scores.size = classify_result->scores.size();
  fd_classify_result_data->scores.data = classify_result->scores.data();
  fd_classify_result_data->type = classify_result->type;
}

FD_DetectionResult* FD_CreateDetectionResult() {
  FD_DetectionResult* fd_detection_result = new FD_DetectionResult();
  fd_detection_result->detection_result =
      std::unique_ptr<fastdeploy::vision::DetectionResult>(
          new fastdeploy::vision::DetectionResult());
  return fd_detection_result;
}

void FD_DestroyDetectionResult(
    __fd_take FD_DetectionResult* fd_detection_result) {
  delete fd_detection_result;
}
}