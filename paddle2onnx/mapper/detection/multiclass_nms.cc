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

#include "paddle2onnx/mapper/detection/multiclass_nms.h"

namespace paddle2onnx {

REGISTER_MAPPER(multiclass_nms3, NMSMapper);

int32_t NMSMapper::GetMinOpset(bool verbose) {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  if (score_info[0].Rank() != 3) {
    Error() << "Lod Tensor input is not supported, which means the shape of "
               "input(scores) is [M, C] now, but Paddle2ONNX only support [N, "
               "C, M]."
            << std::endl;
    return -1;
  }
  if (boxes_info[0].Rank() != 3) {
    Error() << "Only support input boxes as 3-D Tensor, but now it's rank is "
            << boxes_info[0].Rank() << "." << std::endl;
    return -1;
  }
  if (score_info[0].shape[1] <= 0) {
    Error() << "The 2nd-dimension of score should be fixed(means the number of "
               "classes), but now it's "
            << score_info[0].shape[1] << "." << std::endl;
    return -1;
  }

  if (export_as_custom_op || this->deploy_backend == "tensorrt") {
    return 7;
  }

  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void NMSMapper::KeepTopK(const std::string& selected_indices) {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");
  auto value_0 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(0));
  auto value_1 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
  auto value_2 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(2));
  auto value_neg_1 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(-1));

  auto class_id = helper_->MakeNode("Gather", {selected_indices, value_1});
  AddAttribute(class_id, "axis", int64_t(1));

  auto box_id = helper_->MakeNode("Gather", {selected_indices, value_2});
  AddAttribute(box_id, "axis", int64_t(1));

  auto filtered_class_id = class_id->output(0);
  auto filtered_box_id = box_id->output(0);
  if (background_label_ >= 0) {
    auto filter_indices = MapperHelper::Get()->GenName("nms.filter_background");
    auto squeezed_class_id =
        helper_->Squeeze(class_id->output(0), std::vector<int64_t>(1, 1));
    if (background_label_ > 0) {
      auto background = helper_->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, background_label_);
      auto diff = helper_->MakeNode("Sub", {squeezed_class_id, background});
      helper_->MakeNode("NonZero", {diff->output(0)}, {filter_indices});
    } else if (background_label_ == 0) {
      helper_->MakeNode("NonZero", {squeezed_class_id}, {filter_indices});
    }
    auto new_class_id =
        helper_->MakeNode("Gather", {filtered_class_id, filter_indices});
    AddAttribute(new_class_id, "axis", int64_t(0));
    auto new_box_id =
        helper_->MakeNode("Gather", {box_id->output(0), filter_indices});
    AddAttribute(new_box_id, "axis", int64_t(0));
    filtered_class_id = new_class_id->output(0);
    filtered_box_id = new_box_id->output(0);
  }

  // Here is a little complicated
  // Since we need to gather all the scores for the final boxes to filter the
  // top-k boxes Now we have the follow inputs
  //    - scores: [N, C, M] N means batch size(but now it will be regarded as
  //    1); C means number of classes; M means number of boxes for each classes
  //    - selected_indices: [num_selected_indices, 3], and 3 means [batch,
  //    class_id, box_id]. We will use this inputs to gather score
  // So now we will first flatten `scores` to shape of [1 * C * M], then we
  // gather scores by each elements in `selected_indices` The index need be
  // calculated as
  //    `gather_index = class_id * M + box_id`
  auto flatten_score = helper_->Flatten(score_info[0].name);
  auto num_boxes_each_class = helper_->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::INT64, score_info[0].shape[2]);
  auto gather_indices_0 =
      helper_->MakeNode("Mul", {filtered_class_id, num_boxes_each_class});
  auto gather_indices_1 =
      helper_->MakeNode("Add", {gather_indices_0->output(0), filtered_box_id});
  auto gather_indices = helper_->Flatten(gather_indices_1->output(0));
  auto gathered_scores =
      helper_->MakeNode("Gather", {flatten_score, gather_indices});
  AddAttribute(gathered_scores, "axis", int64_t(0));

  // Now we will perform keep_top_k process
  // First we need to check if the number of remaining boxes is greater than
  // keep_top_k Otherwise, we will downgrade the keep_top_k to number of
  // remaining boxes
  auto final_classes = filtered_class_id;
  auto final_boxes_id = filtered_box_id;
  auto final_scores = gathered_scores->output(0);
  if (keep_top_k_ > 0) {
    // get proper topk
    auto shape_of_scores = helper_->MakeNode("Shape", {final_scores});
    auto num_of_boxes =
        helper_->Slice(shape_of_scores->output(0), std::vector<int64_t>(1, 0),
                       std::vector<int64_t>(1, 0), std::vector<int64_t>(1, 1));
    auto top_k =
        helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, keep_top_k_);
    auto ensemble_value = helper_->MakeNode("Concat", {num_of_boxes, top_k});
    AddAttribute(ensemble_value, "axis", int64_t(0));
    auto new_top_k =
        helper_->MakeNode("ReduceMin", {ensemble_value->output(0)});
    AddAttribute(new_top_k, "axes", std::vector<int64_t>(1, 0));
    AddAttribute(new_top_k, "keepdims", int64_t(1));

    // the output is topk_scores, topk_score_indices
    auto topk_node =
        helper_->MakeNode("TopK", {final_scores, new_top_k->output(0)}, 2);
    auto topk_scores =
        helper_->MakeNode("Gather", {final_scores, topk_node->output(1)});
    AddAttribute(topk_scores, "axis", int64_t(0));
    filtered_class_id =
        helper_->MakeNode("Squeeze", {filtered_class_id})->output(0);
    auto topk_classes =
        helper_->MakeNode("Gather", {filtered_class_id, topk_node->output(1)});
    AddAttribute(topk_classes, "axis", int64_t(0));
    filtered_box_id =
        helper_->MakeNode("Squeeze", {filtered_box_id})->output(0);
    auto topk_boxes_id =
        helper_->MakeNode("Gather", {filtered_box_id, topk_node->output(1)});
    AddAttribute(topk_boxes_id, "axis", int64_t(0));

    final_boxes_id = topk_boxes_id->output(0);
    final_scores = topk_scores->output(0);
    final_classes = topk_classes->output(0);
  }

  auto flatten_boxes_id = helper_->Flatten({final_boxes_id});
  auto gathered_selected_boxes =
      helper_->MakeNode("Gather", {boxes_info[0].name, flatten_boxes_id});
  AddAttribute(gathered_selected_boxes, "axis", int64_t(1));

  auto float_classes = helper_->MakeNode("Cast", {final_classes});
  AddAttribute(float_classes, "to", ONNX_NAMESPACE::TensorProto::FLOAT);

  std::vector<int64_t> shape{1, -1, 1};
  auto unsqueezed_scores = helper_->Reshape({final_scores}, shape);

  auto unsqueezed_class = helper_->Reshape({float_classes->output(0)}, shape);

  auto box_result =
      helper_->MakeNode("Concat", {unsqueezed_class, unsqueezed_scores,
                                   gathered_selected_boxes->output(0)});
  AddAttribute(box_result, "axis", int64_t(2));
  helper_->Squeeze({box_result->output(0)}, {out_info[0].name},
                   std::vector<int64_t>(1, 0));

  // other outputs, we don't use sometimes
  // there's lots of Cast in exporting
  // TODO(jiangjiajun) A pass to eleminate all the useless Cast is needed
  auto reshaped_index_result =
      helper_->Reshape({flatten_boxes_id}, {int64_t(-1), int64_t(1)});
  auto index_result =
      helper_->MakeNode("Cast", {reshaped_index_result}, {index_info[0].name});
  AddAttribute(index_result, "to", GetOnnxDtype(index_info[0].dtype));

  auto out_box_shape = helper_->MakeNode("Shape", {out_info[0].name});
  auto num_rois_result =
      helper_->Slice({out_box_shape->output(0)}, std::vector<int64_t>(1, 0),
                     std::vector<int64_t>(1, 0), std::vector<int64_t>(1, 1));
  auto int32_num_rois_result =
      helper_->AutoCast(num_rois_result, num_rois_info[0].name,
                        P2ODataType::INT64, num_rois_info[0].dtype);
}

void NMSMapper::Opset10() {
  if (this->deploy_backend == "tensorrt") {
    return ExportForTensorRT();
  }
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  if (boxes_info[0].shape[0] != 1) {
    Warn()
        << "[WARNING] Due to the operator multiclass_nms3, the exported ONNX "
           "model will only supports inference with input batch_size == 1."
        << std::endl;
  }
  int64_t num_classes = score_info[0].shape[1];
  auto score_threshold = helper_->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::FLOAT, score_threshold_);
  auto nms_threshold = helper_->Constant(
      {1}, ONNX_NAMESPACE::TensorProto::FLOAT, nms_threshold_);
  auto nms_top_k =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, nms_top_k_);

  auto selected_box_index = MapperHelper::Get()->GenName("nms.selected_index");
  if (normalized_) {
    helper_->MakeNode("NonMaxSuppression",
                      {boxes_info[0].name, score_info[0].name, nms_top_k,
                       nms_threshold, score_threshold},
                      {selected_box_index});
  } else {
    auto value_1 =
        helper_->Constant({1}, GetOnnxDtype(boxes_info[0].dtype), float(1.0));
    auto split_boxes = helper_->Split(boxes_info[0].name,
                                      std::vector<int64_t>(4, 1), int64_t(2));
    auto xmax = helper_->MakeNode("Add", {split_boxes[2], value_1});
    auto ymax = helper_->MakeNode("Add", {split_boxes[3], value_1});
    auto new_boxes = helper_->MakeNode(
        "Concat",
        {split_boxes[0], split_boxes[1], xmax->output(0), ymax->output(0)});
    AddAttribute(new_boxes, "axis", int64_t(2));
    helper_->MakeNode("NonMaxSuppression",
                      {new_boxes->output(0), score_info[0].name, nms_top_k,
                       nms_threshold, score_threshold},
                      {selected_box_index});
  }
  KeepTopK(selected_box_index);
}

void NMSMapper::ExportAsCustomOp() {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");
  auto node = helper_->MakeNode(
      custom_op_name, {boxes_info[0].name, score_info[0].name},
      {out_info[0].name, index_info[0].name, num_rois_info[0].name});
  node->set_domain("Paddle");
  int64_t normalized = normalized_ ? 1 : 0;
  AddAttribute(node, "normalized", normalized);
  AddAttribute(node, "nms_threshold", nms_threshold_);
  AddAttribute(node, "score_threshold", score_threshold_);
  AddAttribute(node, "nms_eta", nms_eta_);
  AddAttribute(node, "nms_top_k", nms_top_k_);
  AddAttribute(node, "background_label", background_label_);
  AddAttribute(node, "keep_top_k", keep_top_k_);
  helper_->MakeValueInfo(boxes_info[0].name, boxes_info[0].dtype,
                         boxes_info[0].shape);
  helper_->MakeValueInfo(score_info[0].name, score_info[0].dtype,
                         score_info[0].shape);
  helper_->MakeValueInfo(out_info[0].name, out_info[0].dtype,
                         out_info[0].shape);
  helper_->MakeValueInfo(index_info[0].name, index_info[0].dtype,
                         index_info[0].shape);
  helper_->MakeValueInfo(num_rois_info[0].name, num_rois_info[0].dtype,
                         num_rois_info[0].shape);
}

void NMSMapper::ExportForTensorRT() {
  auto boxes_info = GetInput("BBoxes");
  auto score_info = GetInput("Scores");
  auto out_info = GetOutput("Out");
  auto index_info = GetOutput("Index");
  auto num_rois_info = GetOutput("NmsRoisNum");

  auto scores = helper_->Transpose(score_info[0].name, {0, 2, 1});
  auto boxes = helper_->Unsqueeze(boxes_info[0].name, {2});
  int64_t num_classes = score_info[0].shape[1];
  auto repeats =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                        std::vector<int64_t>({1, 1, num_classes, 1}));
  boxes = helper_->MakeNode("Tile", {boxes, repeats})->output(0);

  auto nms_node =
      helper_->MakeNode("BatchedNMSDynamic_TRT", {boxes, scores}, 4);
  AddAttribute(nms_node, "shareLocation", int64_t(0));
  AddAttribute(nms_node, "backgroundLabelId", background_label_);
  AddAttribute(nms_node, "numClasses", num_classes);
  int64_t nms_top_k = nms_top_k_;
  int64_t keep_top_k = keep_top_k_;
  if (nms_top_k > 4096) {
    Warn()
        << "Paramter nms_top_k:" << nms_top_k
        << " is exceed limit in TensorRT BatchedNMS plugin, will force to 4096."
        << std::endl;
    nms_top_k = 4096;
  }
  if (keep_top_k > 4096) {
    Warn()
        << "Parameter keep_top_k:" << keep_top_k
        << " is exceed limit in TensorRT BatchedNMS plugin, will force to 4096."
        << std::endl;
    keep_top_k = 4096;
  }
  AddAttribute(nms_node, "topK", nms_top_k);
  AddAttribute(nms_node, "keepTopK", keep_top_k);
  AddAttribute(nms_node, "scoreThreshold", score_threshold_);
  AddAttribute(nms_node, "iouThreshold", nms_threshold_);
  if (normalized_) {
    AddAttribute(nms_node, "isNormalized", int64_t(1));
  } else {
    AddAttribute(nms_node, "isNormalized", int64_t(0));
  }
  AddAttribute(nms_node, "clipBoxes", int64_t(0));
  nms_node->set_domain("Paddle");

  auto num_rois = helper_->Reshape(nms_node->output(0), {-1});
  helper_->AutoCast(num_rois, num_rois_info[0].name, P2ODataType::INT32,
                    num_rois_info[0].dtype);

  auto out_classes = helper_->Reshape(nms_node->output(3), {-1, 1});
  auto out_scores = helper_->Reshape(nms_node->output(2), {-1, 1});
  auto out_boxes = helper_->Reshape(nms_node->output(1), {-1, 4});
  out_classes =
      helper_->AutoCast(out_classes, P2ODataType::INT32, P2ODataType::FP32);
  helper_->Concat({out_classes, out_scores, out_boxes}, {out_info[0].name}, 1);

  //  EfficientNMS_TRT cannot get the same result, so disable now
  //  auto nms_node = helper_->MakeNode("EfficientNMS_TRT", {boxes_info[0].name,
  //  score}, 4);
  //  AddAttribute(nms_node, "plugin_version", "1");
  //  AddAttribute(nms_node, "background_class", background_label_);
  //  AddAttribute(nms_node, "max_output_boxes", nms_top_k_);
  //  AddAttribute(nms_node, "score_threshold", score_threshold_);
  //  AddAttribute(nms_node, "iou_threshold", nms_threshold_);
  //  AddAttribute(nms_node, "score_activation", int64_t(0));
  //  AddAttribute(nms_node, "box_coding", int64_t(0));
  //  nms_node->set_domain("Paddle");
  //
  //  auto num_rois = helper_->Reshape(nms_node->output(0), {-1});
  //  helper_->AutoCast(num_rois, num_rois_info[0].name, P2ODataType::INT32,
  //  num_rois_info[0].dtype);
  //
  //  auto out_classes = helper_->Reshape(nms_node->output(3), {-1, 1});
  //  auto out_scores = helper_->Reshape(nms_node->output(2), {-1, 1});
  //  auto out_boxes = helper_->Reshape(nms_node->output(1), {-1, 4});
  //  out_classes = helper_->AutoCast(out_classes, P2ODataType::INT32,
  //  P2ODataType::FP32);
  //  helper_->Concat({out_classes, out_scores, out_boxes}, {out_info[0].name},
  //  1);
}

}  // namespace paddle2onnx
