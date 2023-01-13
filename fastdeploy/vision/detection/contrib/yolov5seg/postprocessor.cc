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

#include "fastdeploy/vision/detection/contrib/yolov5seg/postprocessor.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"
#ifdef ENABLE_PADDLE_BACKEND
#include "paddle/include/experimental/phi/api/include/api.h"
#include "paddle/include/experimental/phi/api/include/tensor.h"
#endif

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv5SegPostprocessor::YOLOv5SegPostprocessor() {
  conf_threshold_ = 0.25;
  nms_threshold_ = 0.5;
  mask_threshold_ = 0.5;
  multi_label_ = true;
  max_wh_ = 7680.0;
  mask_nums_ = 32;
}

bool YOLOv5SegPostprocessor::Run(
    const std::vector<FDTensor>& tensors, std::vector<DetectionResult>* results,
    const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {
  int batch = tensors[0].shape[0];

  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    // store mask information
    std::vector<std::vector<float>> mask_embeddings;
    (*results)[bs].Clear();
    if (multi_label_) {
      (*results)[bs].Reserve(tensors[0].shape[1] *
                             (tensors[0].shape[2] - mask_nums_ - 5));
    } else {
      (*results)[bs].Reserve(tensors[0].shape[1]);
    }
    if (tensors[0].dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    const float* data = reinterpret_cast<const float*>(tensors[0].Data()) +
                        bs * tensors[0].shape[1] * tensors[0].shape[2];
    for (size_t i = 0; i < tensors[0].shape[1]; ++i) {
      int s = i * tensors[0].shape[2];
      float cls_conf = data[s + 4];
      float confidence = data[s + 4];
      if (multi_label_) {
        for (size_t j = 5; j < tensors[0].shape[2] - mask_nums_; ++j) {
          confidence = data[s + 4];
          const float* class_score = data + s + j;
          confidence *= (*class_score);
          // filter boxes by conf_threshold
          if (confidence <= conf_threshold_) {
            continue;
          }
          int32_t label_id = std::distance(data + s + 5, class_score);
          // for mask_embedding
          std::vector<float> mask_embedding(
              data + s + tensors[0].shape[2] - mask_nums_,
              data + s + tensors[0].shape[2]);
          for (size_t k = 0; k < mask_embedding.size(); ++k) {
            mask_embedding[k] *= cls_conf;
          }

          // convert from [x, y, w, h] to [x1, y1, x2, y2]
          (*results)[bs].boxes.emplace_back(std::array<float, 4>{
              data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
              data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
          (*results)[bs].label_ids.push_back(label_id);
          (*results)[bs].scores.push_back(confidence);
          // TODO(wangjunjie06): No zero copy
          mask_embeddings.push_back(mask_embedding);
        }
      } else {
        const float* max_class_score = std::max_element(
            data + s + 5, data + s + tensors[0].shape[2] - mask_nums_);
        confidence *= (*max_class_score);
        // filter boxes by conf_threshold
        if (confidence <= conf_threshold_) {
          continue;
        }
        int32_t label_id = std::distance(data + s + 5, max_class_score);
        // for mask_embedding
        std::vector<float> mask_embedding(
            data + s + tensors[0].shape[2] - mask_nums_,
            data + s + tensors[0].shape[2]);
        for (size_t k = 0; k < mask_embedding.size(); ++k) {
          mask_embedding[k] *= cls_conf;
        }
        // convert from [x, y, w, h] to [x1, y1, x2, y2]
        (*results)[bs].boxes.emplace_back(std::array<float, 4>{
            data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
            data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
        (*results)[bs].label_ids.push_back(label_id);
        (*results)[bs].scores.push_back(confidence);
        mask_embeddings.push_back(mask_embedding);
      }
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }
    // get box index after nms
    std::vector<int> index;
    utils::NMS(&((*results)[bs]), nms_threshold_, &index);

    // deal with mask
    // step1: MatMul, (box_nums * 32) x (32 * 160 * 160) = box_nums * 160 * 160
    // step2: Sigmoid
    // step3: Resize to original image size
    // step4: Select pixels greater than threshold and crop
    (*results)[bs].contain_masks = true;
    (*results)[bs].masks.resize((*results)[bs].boxes.size());
    const float* data_mask =
        reinterpret_cast<const float*>(tensors[1].Data()) +
        bs * tensors[1].shape[1] * tensors[1].shape[2] * tensors[1].shape[3];
    cv::Mat mask_proto =
        cv::Mat(tensors[1].shape[1], tensors[1].shape[2] * tensors[1].shape[3],
                CV_32FC(1), const_cast<float*>(data_mask));
    // vector to cv::Mat for MatMul
    // after push_back, Mat of m*n becomes (m + 1) * n
    cv::Mat mask_proposals;
    for (size_t i = 0; i < index.size(); ++i) {
      mask_proposals.push_back(cv::Mat(mask_embeddings[index[i]]).t());
    }
    cv::Mat matmul_result = (mask_proposals * mask_proto).t();
    cv::Mat masks = matmul_result.reshape(
        (*results)[bs].boxes.size(), {static_cast<int>(tensors[1].shape[2]),
                                      static_cast<int>(tensors[1].shape[3])});
    // split for boxes nums
    std::vector<cv::Mat> mask_channels;
    cv::split(masks, mask_channels);

    // scale the boxes to the origin image shape
    auto iter_out = ims_info[bs].find("output_shape");
    auto iter_ipt = ims_info[bs].find("input_shape");
    FDASSERT(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end(),
             "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float scale = std::min(out_h / ipt_h, out_w / ipt_w);
    float pad_h = (out_h - ipt_h * scale) / 2;
    float pad_w = (out_w - ipt_w * scale) / 2;
    // for mask
    float pad_h_mask = (float)pad_h / out_h * tensors[1].shape[2];
    float pad_w_mask = (float)pad_w / out_w * tensors[1].shape[3];
#ifdef ENABLE_PADDLE_BACKEND
    // convert vector to paddle::Tensor for Matmul
    std::vector<int64_t> tensor0_shape{
        static_cast<int64_t>(index.size()),
        static_cast<int64_t>(mask_embeddings[0].size())};
    // tc.Start();
    auto mask_proposals = paddle::experimental::empty(
        tensor0_shape, paddle::experimental::DataType::FLOAT32,
        phi::CPUPlace());
    float* mask_proposals_data = mask_proposals.data<float>();
    size_t numel0 = mask_embeddings[0].size();
    for (size_t i = 0; i < index.size(); ++i) {
      memcpy(mask_proposals_data + i * numel0,
             static_cast<float*>(mask_embeddings[index[i]].data()),
             numel0 * sizeof(float));
    }
    // tc.End();
    // std::cout << "time tensor0 copy = " << tc.Duration() * 1000 << "ms" <<
    // std::endl;
    std::vector<int64_t> tensor1_shape{
        tensors[1].shape[1], tensors[1].shape[2] * tensors[1].shape[3]};
    // tc.Start();
    auto mask_proto = paddle::experimental::empty(
        tensor1_shape, paddle::experimental::DataType::FLOAT32,
        phi::CPUPlace());
    size_t numel1 =
        tensors[1].shape[1] * tensors[1].shape[2] * tensors[1].shape[3];
    float* mask_proto_data = mask_proto.data<float>();
    memcpy(mask_proto_data, const_cast<float*>(data_mask),
           numel1 * sizeof(float));
    // tc.End();
    // std::cout << "time tensor1 copy = " << tc.Duration() * 1000 << "ms" <<
    // std::endl;
    // tc.Start();
    auto matmul_result =
        paddle::experimental::matmul(mask_proposals, mask_proto);
    // tc.End();
    // std::cout << "time matmul = " << tc.Duration() * 1000 << "ms" <<
    // std::endl;
    // tc.Start();
    // matmul_result = paddle::experimental::sigmoid(matmul_result);
    // matmul_result.reshape({1, static_cast<int64_t>(index.size()),
    // tensors[1].shape[2], tensors[1].shape[3]});
    matmul_result = paddle::experimental::reshape(
        matmul_result, {1, static_cast<int64_t>(index.size()),
                        tensors[1].shape[2], tensors[1].shape[3]});
    // tc.End();
    // std::cout << "time reshape = " << tc.Duration() * 1000 << "ms" <<
    // std::endl;
    // crop mask for feature map
    int x1 = static_cast<int>(pad_w_mask);
    int y1 = static_cast<int>(pad_h_mask);
    int x2 = static_cast<int>(tensors[1].shape[3] - pad_w_mask);
    int y2 = static_cast<int>(tensors[1].shape[2] - pad_h_mask);
    // tc.Start();
    auto crop_result = paddle::experimental::crop(
        matmul_result, {1, -1, y2 - y1, x2 - x1}, {0, 0, y1, x1});
    crop_result = paddle::experimental::sigmoid(crop_result);
    // tc.End();
    // std::cout << "time crop + sigmoid = " << tc.Duration() * 1000 << "ms" <<
    // std::endl;
    // tc.Start();
    auto resize_result = paddle::experimental::bilinear_interp(
        crop_result, paddle::none, paddle::none, paddle::none, "NCHW", 0, ipt_h,
        ipt_w, {}, "bilinear", false, 0);
    resize_result = paddle::experimental::squeeze(resize_result, {0});
#else
    fastdeploy::TimeCounter tc;
    cv::Mat mask_proto =
        cv::Mat(tensors[1].shape[1], tensors[1].shape[2] * tensors[1].shape[3],
                CV_32FC(1), const_cast<float*>(data_mask));
    // convert vector to cv::Mat for Matmul
    cv::Mat mask_proposals;
    for (size_t i = 0; i < index.size(); ++i) {
      mask_proposals.push_back(cv::Mat(mask_embeddings[index[i]]).t());
    }
    cv::Mat matmul_result = (mask_proposals * mask_proto).t();
    cv::Mat masks = matmul_result.reshape(
        (*results)[bs].boxes.size(), {static_cast<int>(tensors[1].shape[2]),
                                      static_cast<int>(tensors[1].shape[3])});
    // split for boxes nums
    std::vector<cv::Mat> mask_channels;
    cv::split(masks, mask_channels);
#endif
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      int32_t label_id = ((*results)[bs].label_ids)[i];
      // clip box
      (*results)[bs].boxes[i][0] =
          (*results)[bs].boxes[i][0] - max_wh_ * label_id;
      (*results)[bs].boxes[i][1] =
          (*results)[bs].boxes[i][1] - max_wh_ * label_id;
      (*results)[bs].boxes[i][2] =
          (*results)[bs].boxes[i][2] - max_wh_ * label_id;
      (*results)[bs].boxes[i][3] =
          (*results)[bs].boxes[i][3] - max_wh_ * label_id;
      (*results)[bs].boxes[i][0] =
          std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][1] =
          std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][2] =
          std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][3] =
          std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w);
      (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h);
      (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w);
      (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h);
#ifdef ENABLE_PADDLE_BACKEND
      // crop mask for source img
      int x1_src = static_cast<int>(round((*results)[bs].boxes[i][0]));
      int y1_src = static_cast<int>(round((*results)[bs].boxes[i][1]));
      int x2_src = static_cast<int>(round((*results)[bs].boxes[i][2]));
      int y2_src = static_cast<int>(round((*results)[bs].boxes[i][3]));
      // tc.Start();
      auto mask = paddle::experimental::crop(
          resize_result, {1, y2_src - y1_src, x2_src - x1_src},
          {static_cast<int>(i), y1_src, x1_src});
      auto mask_threshold = paddle::experimental::full(
          mask.shape(), mask_threshold_, mask.dtype(), mask.place());
      auto mask_result =
          paddle::experimental::greater_than(mask, mask_threshold);
      // tc.End();
      // std::cout << "time crop + greater_than = " << tc.Duration() * 1000 <<
      // "ms" << std::endl;
      // save mask in DetectionResult
      int keep_mask_h = y2_src - y1_src;
      int keep_mask_w = x2_src - x1_src;
      int keep_mask_numel = keep_mask_h * keep_mask_w;
      (*results)[bs].masks[i].Resize(keep_mask_numel);
      (*results)[bs].masks[i].shape = {keep_mask_h, keep_mask_w};
      uint8_t* keep_mask_ptr =
          reinterpret_cast<uint8_t*>((*results)[bs].masks[i].Data());
      std::memcpy(keep_mask_ptr,
                  reinterpret_cast<uint8_t*>(mask_result.data<bool>()),
                  keep_mask_numel * sizeof(uint8_t));
#else
      // deal with mask
      cv::Mat dest, mask;
      // sigmoid
      cv::exp(-mask_channels[i], dest);
      dest = 1.0 / (1.0 + dest);
      // crop mask for feature map
      int x1 = static_cast<int>(pad_w_mask);
      int y1 = static_cast<int>(pad_h_mask);
      int x2 = static_cast<int>(tensors[1].shape[3] - pad_w_mask);
      int y2 = static_cast<int>(tensors[1].shape[2] - pad_h_mask);
      cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
      dest = dest(roi);
      cv::resize(dest, mask, cv::Size(ipt_w, ipt_h), 0, 0, cv::INTER_LINEAR);
      // crop mask for source img
      int x1_src = static_cast<int>(round((*results)[bs].boxes[i][0]));
      int y1_src = static_cast<int>(round((*results)[bs].boxes[i][1]));
      int x2_src = static_cast<int>(round((*results)[bs].boxes[i][2]));
      int y2_src = static_cast<int>(round((*results)[bs].boxes[i][3]));
      cv::Rect roi_src(x1_src, y1_src, x2_src - x1_src, y2_src - y1_src);
      mask = mask(roi_src);
      mask = mask > mask_threshold_;
      // save mask in DetectionResult
      int keep_mask_h = y2_src - y1_src;
      int keep_mask_w = x2_src - x1_src;
      int keep_mask_numel = keep_mask_h * keep_mask_w;
      (*results)[bs].masks[i].Resize(keep_mask_numel);
      (*results)[bs].masks[i].shape = {keep_mask_h, keep_mask_w};
      uint8_t* keep_mask_ptr =
          reinterpret_cast<uint8_t*>((*results)[bs].masks[i].Data());
      std::memcpy(keep_mask_ptr, reinterpret_cast<uint8_t*>(mask.ptr()),
                  keep_mask_numel * sizeof(uint8_t));
#endif
    }
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
