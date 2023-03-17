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

#include "paddle2onnx/mapper/nn/softmax_with_cross_entropy.h"

namespace paddle2onnx {
REGISTER_MAPPER(softmax_with_cross_entropy, SoftmaxCrossEntropyLossMapper)

int32_t SoftmaxCrossEntropyLossMapper::GetMinOpset(bool verbose) {
  auto logits = GetInput("Logits");
  std::vector<int64_t> logits_shape = logits[0].shape;
  if (logits_shape.size() < 2) {
    Error() << "SoftmaxCrossEntropyLoss in onnx not support 1D logits."
            << std::endl;
    return -1;
  }
  Logger(verbose, 12) << RequireOpset(12) << std::endl;
  return 12;
}

void SoftmaxCrossEntropyLossMapper::Opset12() {
  auto logits = GetInput("Logits");
  auto labels = GetInput("Label");

  auto loss = GetOutput("Loss");
  auto softmax = GetOutput("Softmax");
  std::vector<int64_t> logits_shape = logits[0].shape;
  auto dim = logits[0].Rank();
  if (axis_ < 0) {
    axis_ += dim;
  }
  if (soft_label_) {
    std::vector<int64_t> split;
    split.resize(logits_shape[axis_], 1);
    std::vector<int64_t> axes_val = {axis_};
    std::string axes_node =
        helper_->Constant(GetOnnxDtype(P2ODataType::INT64), axes_val);
    if (axis_ == dim - 1) {
      auto logsoftmax_node = helper_->MakeNode("LogSoftmax", {logits[0].name});
      AddAttribute(logsoftmax_node, "axis", axis_);
      helper_->MakeNode("Exp", {logsoftmax_node->output(0)}, {softmax[0].name});
      auto mul_result = helper_->MakeNode(
          "Mul", {logsoftmax_node->output(0), labels[0].name});
      if (helper_->GetOpsetVersion() < 13) {
        auto reducesum_node =
            helper_->MakeNode("ReduceSum", {mul_result->output(0)});
        AddAttribute(reducesum_node, "axes", axes_val);
        helper_->MakeNode("Neg", {reducesum_node->output(0)}, {loss[0].name});
      } else {
        auto reducesum_node =
            helper_->MakeNode("ReduceSum", {mul_result->output(0), axes_node});
        helper_->MakeNode("Neg", {reducesum_node->output(0)}, {loss[0].name});
      }
    } else {
      auto perm = Arange(0, dim);
      perm[dim - 1] = axis_;
      perm[axis_] = dim - 1;
      auto output = helper_->Transpose(logits[0].name, perm);
      auto logsoftmax_node = helper_->MakeNode("LogSoftmax", {output});
      AddAttribute(logsoftmax_node, "axis", int64_t(-1));
      auto transpose_logsoftmax_node =
          helper_->Transpose(logsoftmax_node->output(0), perm);
      helper_->MakeNode("Exp", {transpose_logsoftmax_node}, {softmax[0].name});
      auto mul_result =
          helper_->MakeNode("Mul", {transpose_logsoftmax_node, labels[0].name});
      if (helper_->GetOpsetVersion() < 13) {
        auto reducesum_node =
            helper_->MakeNode("ReduceSum", {mul_result->output(0)});
        AddAttribute(reducesum_node, "axes", axes_val);
        helper_->MakeNode("Neg", {reducesum_node->output(0)}, {loss[0].name});
      } else {
        auto reducesum_node =
            helper_->MakeNode("ReduceSum", {mul_result->output(0), axes_node});
        helper_->MakeNode("Neg", {reducesum_node->output(0)}, {loss[0].name});
      }
    }
  } else {
    if (axis_ == 1) {
      auto squeeze_node = helper_->Squeeze(labels[0].name, {axis_});
      auto node = helper_->MakeNode("SoftmaxCrossEntropyLoss",
                                    {logits[0].name, squeeze_node}, 2);
      AddAttribute(node, "ignore_index", ignore_index_);
      AddAttribute(node, "reduction", "none");
      auto loss_node =
          helper_->Unsqueeze(node->output(0), loss[0].name, {axis_});
      // onnx output is log(softmax), but paddle output is softmax
      helper_->MakeNode("Exp", {node->output(1)}, {softmax[0].name});
    } else {
      std::vector<int64_t> perm = Arange(0, dim);
      perm[1] = axis_;
      perm[axis_] = 1;
      auto transpose_logits = helper_->MakeNode("Transpose", {logits[0].name});
      AddAttribute(transpose_logits, "perm", perm);
      auto transpose_labels = helper_->MakeNode("Transpose", {labels[0].name});
      AddAttribute(transpose_labels, "perm", perm);
      auto squeeze_labels = helper_->Squeeze(transpose_labels->output(0), {1});
      auto node =
          helper_->MakeNode("SoftmaxCrossEntropyLoss",
                            {transpose_logits->output(0), squeeze_labels}, 2);
      AddAttribute(node, "ignore_index", ignore_index_);
      AddAttribute(node, "reduction", "none");
      auto unsqueeze_node = helper_->Unsqueeze(node->output(0), {1});
      auto revert_transpose_logits =
          helper_->MakeNode("Transpose", {unsqueeze_node}, {loss[0].name});
      AddAttribute(revert_transpose_logits, "perm", perm);
      auto revert_transpose_softmax =
          helper_->MakeNode("Transpose", {node->output(1)});
      AddAttribute(revert_transpose_softmax, "perm", perm);
      // onnx output is log(softmax), but paddle output is softmax
      helper_->MakeNode("Exp", {revert_transpose_softmax->output(0)},
                        {softmax[0].name});
    }
  }
}
}  // namespace paddle2onnx
