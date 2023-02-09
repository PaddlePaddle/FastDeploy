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

extern "C" {

// Classification Results

FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapper() {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      new FD_C_ClassifyResultWrapper();
  fd_c_classify_result_wrapper->classify_result =
      std::unique_ptr<fastdeploy::vision::ClassifyResult>(
          new fastdeploy::vision::ClassifyResult());
  return fd_c_classify_result_wrapper;
}

void FD_C_DestroyClassifyResultWrapper(
    __fd_take FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper) {
  delete fd_c_classify_result_wrapper;
}

void FD_C_DestroyClassifyResult(
    __fd_take FD_C_ClassifyResult* fd_c_classify_result) {
  if (fd_c_classify_result == nullptr) return;
  // delete label_ids
  delete[] fd_c_classify_result->label_ids.data;
  // delete scores
  delete[] fd_c_classify_result->scores.data;
  delete fd_c_classify_result;
}

FD_C_ClassifyResult* FD_C_ClassifyResultWrapperGetData(
    __fd_keep FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper) {
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  FD_C_ClassifyResult* fd_c_classify_result_data = new FD_C_ClassifyResult();
  // copy label_ids
  fd_c_classify_result_data->label_ids.size = classify_result->label_ids.size();
  fd_c_classify_result_data->label_ids.data =
      new int32_t[fd_c_classify_result_data->label_ids.size];
  memcpy(fd_c_classify_result_data->label_ids.data,
         classify_result->label_ids.data(),
         sizeof(int32_t) * fd_c_classify_result_data->label_ids.size);
  // copy scores
  fd_c_classify_result_data->scores.size = classify_result->scores.size();
  fd_c_classify_result_data->scores.data =
      new float[fd_c_classify_result_data->scores.size];
  memcpy(fd_c_classify_result_data->scores.data, classify_result->scores.data(),
         sizeof(float) * fd_c_classify_result_data->scores.size);
  fd_c_classify_result_data->type =
      static_cast<FD_C_ResultType>(classify_result->type);
  return fd_c_classify_result_data;
}

FD_C_ClassifyResultWrapper* FD_C_CreateClassifyResultWrapperFromData(
    __fd_keep FD_C_ClassifyResult* fd_c_classify_result) {
  FD_C_ClassifyResultWrapper* fd_c_classify_result_wrapper =
      FD_C_CreateClassifyResultWrapper();
  auto& classify_result = CHECK_AND_CONVERT_FD_TYPE(
      ClassifyResultWrapper, fd_c_classify_result_wrapper);
  // copy label_ids
  classify_result->label_ids.resize(fd_c_classify_result->label_ids.size);
  memcpy(classify_result->label_ids.data(),
         fd_c_classify_result->label_ids.data,
         sizeof(int32_t) * fd_c_classify_result->label_ids.size);
  // copy scores
  classify_result->scores.resize(fd_c_classify_result->scores.size);
  memcpy(classify_result->scores.data(), fd_c_classify_result->scores.data,
         sizeof(int32_t) * fd_c_classify_result->scores.size);
  classify_result->type =
      static_cast<fastdeploy::vision::ResultType>(fd_c_classify_result->type);
  return fd_c_classify_result_wrapper;
}

// Detection Results

FD_C_DetectionResultWrapper* FD_C_CreateDetectionResultWrapper() {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      new FD_C_DetectionResultWrapper();
  fd_c_detection_result_wrapper->detection_result =
      std::unique_ptr<fastdeploy::vision::DetectionResult>(
          new fastdeploy::vision::DetectionResult());
  return fd_c_detection_result_wrapper;
}

void FD_C_DestroyDetectionResultWrapper(
    __fd_take FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  delete fd_c_detection_result_wrapper;
}

void FD_C_DestroyDetectionResult(
    __fd_take FD_C_DetectionResult* fd_c_detection_result) {
  if (fd_c_detection_result == nullptr) return;
  // delete boxes
  for (size_t i = 0; i < fd_c_detection_result->boxes.size; i++) {
    delete[] fd_c_detection_result->boxes.data[i].data;
  }
  delete[] fd_c_detection_result->boxes.data;
  // delete scores
  delete[] fd_c_detection_result->scores.data;
  // delete label_ids
  delete[] fd_c_detection_result->label_ids.data;
  // delete masks
  for (size_t i = 0; i < fd_c_detection_result->masks.size; i++) {
    delete[] fd_c_detection_result->masks.data[i].data.data;
    delete[] fd_c_detection_result->masks.data[i].shape.data;
  }
  delete fd_c_detection_result;
}

FD_C_DetectionResult* FD_C_DetectionResultWrapperGetData(
    __fd_keep FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  FD_C_DetectionResult* fd_c_detection_result = new FD_C_DetectionResult();
  // copy boxes
  const int boxes_coordinate_dim = 4;
  fd_c_detection_result->boxes.size = detection_result->boxes.size();
  fd_c_detection_result->boxes.data =
      new FD_C_OneDimArrayFloat[fd_c_detection_result->boxes.size];
  for (size_t i = 0; i < detection_result->boxes.size(); i++) {
    fd_c_detection_result->boxes.data[i].size = boxes_coordinate_dim;
    fd_c_detection_result->boxes.data[i].data = new float[boxes_coordinate_dim];
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      fd_c_detection_result->boxes.data[i].data[j] =
          detection_result->boxes[i][j];
    }
  }
  // copy scores
  fd_c_detection_result->scores.size = detection_result->scores.size();
  fd_c_detection_result->scores.data =
      new float[fd_c_detection_result->scores.size];
  memcpy(fd_c_detection_result->scores.data, detection_result->scores.data(),
         sizeof(float) * fd_c_detection_result->scores.size);
  // copy label_ids
  fd_c_detection_result->label_ids.size = detection_result->label_ids.size();
  fd_c_detection_result->label_ids.data =
      new int32_t[fd_c_detection_result->label_ids.size];
  memcpy(fd_c_detection_result->label_ids.data,
         detection_result->label_ids.data(),
         sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // copy masks
  fd_c_detection_result->masks.size = detection_result->masks.size();
  fd_c_detection_result->masks.data =
      new FD_C_Mask[fd_c_detection_result->masks.size];
  for (size_t i = 0; i < detection_result->masks.size(); i++) {
    // copy data in mask
    fd_c_detection_result->masks.data[i].data.size =
        detection_result->masks[i].data.size();
    fd_c_detection_result->masks.data[i].data.data =
        new uint8_t[detection_result->masks[i].data.size()];
    memcpy(fd_c_detection_result->masks.data[i].data.data,
           detection_result->masks[i].data.data(),
           sizeof(uint8_t) * detection_result->masks[i].data.size());
    // copy shape in mask
    fd_c_detection_result->masks.data[i].shape.size =
        detection_result->masks[i].shape.size();
    fd_c_detection_result->masks.data[i].shape.data =
        new int64_t[detection_result->masks[i].shape.size()];
    memcpy(fd_c_detection_result->masks.data[i].shape.data,
           detection_result->masks[i].shape.data(),
           sizeof(int64_t) * detection_result->masks[i].shape.size());
    fd_c_detection_result->masks.data[i].type =
        static_cast<FD_C_ResultType>(detection_result->masks[i].type);
  }
  fd_c_detection_result->contain_masks = detection_result->contain_masks;
  fd_c_detection_result->type =
      static_cast<FD_C_ResultType>(detection_result->type);
  return fd_c_detection_result;
}

FD_C_DetectionResultWrapper* FD_C_CreateDetectionResultWrapperFromData(
    __fd_keep FD_C_DetectionResult* fd_c_detection_result) {
  FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper =
      FD_C_CreateDetectionResultWrapper();
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);

  // copy boxes
  const int boxes_coordinate_dim = 4;
  detection_result->boxes.resize(fd_c_detection_result->boxes.size);
  for (size_t i = 0; i < fd_c_detection_result->boxes.size; i++) {
    for (size_t j = 0; j < boxes_coordinate_dim; j++) {
      detection_result->boxes[i][j] =
          fd_c_detection_result->boxes.data[i].data[j];
    }
  }
  // copy scores
  detection_result->scores.resize(fd_c_detection_result->scores.size);
  memcpy(detection_result->scores.data(), fd_c_detection_result->scores.data,
         sizeof(float) * fd_c_detection_result->scores.size);
  // copy label_ids
  detection_result->label_ids.resize(fd_c_detection_result->label_ids.size);
  memcpy(detection_result->label_ids.data(),
         fd_c_detection_result->label_ids.data,
         sizeof(int32_t) * fd_c_detection_result->label_ids.size);
  // copy masks
  detection_result->masks.resize(fd_c_detection_result->masks.size);
  for (size_t i = 0; i < fd_c_detection_result->masks.size; i++) {
    // copy data in mask
    detection_result->masks[i].data.resize(
        fd_c_detection_result->masks.data[i].data.size);
    memcpy(detection_result->masks[i].data.data(),
           fd_c_detection_result->masks.data[i].data.data,
           sizeof(uint8_t) * fd_c_detection_result->masks.data[i].data.size);
    // copy shape in mask
    detection_result->masks[i].shape.resize(
        fd_c_detection_result->masks.data[i].shape.size);
    memcpy(detection_result->masks[i].shape.data(),
           fd_c_detection_result->masks.data[i].shape.data,
           sizeof(int64_t) * fd_c_detection_result->masks.data[i].shape.size);
    detection_result->masks[i].type =
        static_cast<fastdeploy::vision::ResultType>(
            fd_c_detection_result->masks.data[i].type);
  }
  detection_result->contain_masks = fd_c_detection_result->contain_masks;
  detection_result->type =
      static_cast<fastdeploy::vision::ResultType>(fd_c_detection_result->type);

  return fd_c_detection_result_wrapper;
}
}