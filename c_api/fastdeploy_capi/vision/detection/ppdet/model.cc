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

#include "fastdeploy_capi/vision/detection/ppdet/model.h"

#include "fastdeploy_capi/types_internal.h"
#include "fastdeploy_capi/vision/visualize.h"

extern "C" {

FD_C_PPYOLOEWrapper* FD_C_CreatesPPYOLOEWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper = new FD_C_PPYOLOEWrapper();
  fd_c_ppyoloe_wrapper->ppyoloe_model =
      std::unique_ptr<fastdeploy::vision::detection::PPYOLOE>(
          new fastdeploy::vision::detection::PPYOLOE(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_ppyoloe_wrapper;
}

void FD_C_DestroyPPYOLOEWrapper(FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper) {
  delete fd_c_ppyoloe_wrapper;
}

FD_C_Bool FD_C_PPYOLOEWrapperPredict(
    FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper, FD_C_Mat img,
    FD_C_DetectionResultWrapper* fd_c_detection_result_wrapper) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& ppyoloe_model =
      CHECK_AND_CONVERT_FD_TYPE(PPYOLOEWrapper, fd_c_ppyoloe_wrapper);
  auto& detection_result = CHECK_AND_CONVERT_FD_TYPE(
      DetectionResultWrapper, fd_c_detection_result_wrapper);
  return ppyoloe_model->Predict(im, detection_result.get());
}

FD_C_Bool FD_C_PPYOLOEWrapperInitialized(
    FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper) {
  return fd_c_ppyoloe_wrapper->Initialized();
}

FD_C_DetectionResult* FD_C_DetectionResultToC(
    fastdeploy::vision::DetectionResult* detection_result) {
  // Internal use, transfer fastdeploy::vision::DetectionResult to
  // FD_C_DetectionResult
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

FD_C_Bool FD_C_PPYOLOEWrapperBatchPredict(
    FD_C_PPYOLOEWrapper* fd_c_ppyoloe_wrapper, FD_C_OneDimMat imgs,
    FD_C_OneDimDetectionResult* results) {
  std::vector<cv::Mat> imgs_vec;
  std::vector<fastdeploy::vision::DetectionResult> results_out;
  for (int i = 0; i < imgs.size; i++) {
    imgs_vec.push_back(*(reinterpret_cast<cv::Mat*>(imgs.data[i])));
  }
  bool successful = fd_c_ppyoloe_wrapper->BatchPredict(imgs_vec, &results_out);
  if (successful) {
    // copy results back to FD_C_OneDimClassifyResult
    results->size = results_out.size();
    results->data = new FD_C_DetectionResult[results->size];
    for (int i = 0; i < results_out.size(); i++) {
      results->data[i] = FD_C_DetectionResultToC(&results_out[i]);
    }
  }
  return successful;
}
}