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

#include "paddle2onnx/mapper/detection/roi_align.h"

namespace paddle2onnx {
REGISTER_MAPPER(roi_align, RoiAlignMapper)

void RoiAlignMapper::Opset10() {
  auto x_info = GetInput("X");
  auto rois_info = GetInput("ROIs");
  auto out_info = GetOutput("Out");

  auto roi_shape = helper_->MakeNode("Shape", {rois_info[0].name})->output(0);
  auto num_rois =
      helper_->Slice(roi_shape, std::vector<int64_t>(1, 0),
                     std::vector<int64_t>(1, 0), std::vector<int64_t>(1, 1));
  auto value_zero = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                      std::vector<int64_t>(1, 0));
  auto batch_indices =
      helper_->MakeNode("Expand", {value_zero, num_rois})->output(0);
  auto roi_align_node = helper_->MakeNode(
      "RoiAlign", {x_info[0].name, rois_info[0].name, batch_indices},
      {out_info[0].name});
  AddAttribute(roi_align_node, "output_height", pooled_height_);
  AddAttribute(roi_align_node, "output_width", pooled_width_);
  AddAttribute(roi_align_node, "sampling_ratio", sampling_ratio_);
  AddAttribute(roi_align_node, "spatial_scale", spatial_scale_);
  AddAttribute(roi_align_node, "mode", "avg");
}

}  // namespace paddle2onnx
