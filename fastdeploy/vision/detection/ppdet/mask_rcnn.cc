// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision/detection/ppdet/mask_rcnn.h"

namespace fastdeploy {
namespace vision {
namespace detection {

MaskRCNN::MaskRCNN(const std::string& model_file,
                   const std::string& params_file,
                   const std::string& config_file,
                   const RuntimeOption& custom_option,
                   const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool MaskRCNN::Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result) {
  // index 0: bbox_data [N, 6] float32
  // index 1: bbox_num [B=1] int32
  // index 2: mask_data [N, h, w] int32
  FDASSERT(infer_result[1].shape[0] == 1,
           "Only support batch = 1 in FastDeploy now.");
  FDASSERT(infer_result.size() == 3,
           "The infer_result must contains 3 otuput Tensors, but found %lu",
           infer_result.size());

  FDTensor& box_tensor = infer_result[0];
  FDTensor& box_num_tensor = infer_result[1];
  FDTensor& mask_tensor = infer_result[2];

  int box_num = 0;
  if (box_num_tensor.dtype == FDDataType::INT32) {
    box_num = *(static_cast<int32_t*>(box_num_tensor.Data()));
  } else if (box_num_tensor.dtype == FDDataType::INT64) {
    box_num = *(static_cast<int64_t*>(box_num_tensor.Data()));
  } else {
    FDASSERT(false,
             "The output box_num of PaddleDetection/MaskRCNN model should be "
             "type of int32/int64.");
  }
  if (box_num <= 0) {
    return true;  // no object detected.
  }
  result->Resize(box_num);
  float* box_data = static_cast<float*>(box_tensor.Data());
  for (size_t i = 0; i < box_num; ++i) {
    result->label_ids[i] = static_cast<int>(box_data[i * 6]);
    result->scores[i] = box_data[i * 6 + 1];
    result->boxes[i] =
        std::array<float, 4>{box_data[i * 6 + 2], box_data[i * 6 + 3],
                             box_data[i * 6 + 4], box_data[i * 6 + 5]};
  }
  result->contain_masks = true;
  // TODO(qiuyanjun): Cast int64/int8 to int32.
  FDASSERT(mask_tensor.dtype == FDDataType::INT32,
           "The dtype of mask Tensor must be int32 now!");
  // In PaddleDetection/MaskRCNN, the mask_h and mask_w
  // are already aligned with original input image. So,
  // we need to crop it from output mask according to
  // the detected bounding box.
  //   +-----------------------+
  //   |  x1,y1                |
  //   |  +---------------+    |
  //   |  |               |    |
  //   |  |      Crop     |    |
  //   |  |               |    |
  //   |  |               |    |
  //   |  +---------------+    |
  //   |                x2,y2  |
  //   +-----------------------+
  int64_t out_mask_h = mask_tensor.shape[1];
  int64_t out_mask_w = mask_tensor.shape[2];
  int64_t out_mask_numel = out_mask_h * out_mask_w;
  int32_t* out_mask_data = static_cast<int32_t*>(mask_tensor.Data());
  for (size_t i = 0; i < box_num; ++i) {
    // crop instance mask according to box
    int64_t x1 = static_cast<int64_t>(result->boxes[i][0]);
    int64_t y1 = static_cast<int64_t>(result->boxes[i][1]);
    int64_t x2 = static_cast<int64_t>(result->boxes[i][2]);
    int64_t y2 = static_cast<int64_t>(result->boxes[i][3]);
    int64_t keep_mask_h = y2 - y1;
    int64_t keep_mask_w = x2 - x1;
    int64_t keep_mask_numel = keep_mask_h * keep_mask_w;
    result->masks[i].Resize(keep_mask_numel);  // int32
    result->masks[i].shape = {keep_mask_h, keep_mask_w};
    int32_t* mask_start_ptr = out_mask_data + i * out_mask_numel;
    int32_t* keep_mask_ptr = static_cast<int32_t*>(result->masks[i].Data());
    for (size_t row = y1; row < y2; ++row) {
      size_t keep_nbytes_in_col = keep_mask_w * sizeof(int32_t);
      int32_t* out_row_start_ptr = mask_start_ptr + row * out_mask_w + x1;
      int32_t* keep_row_start_ptr = keep_mask_ptr + (row - y1) * keep_mask_w;
      std::memcpy(keep_row_start_ptr, out_row_start_ptr, keep_nbytes_in_col);
    }
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
