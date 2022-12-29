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
#include "fastdeploy/vision/utils/utils.h"

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

bool YOLOv5SegPostprocessor::Run(const std::vector<FDTensor>& tensors, std::vector<DetectionResult>* results,
                              const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {
  int batch = tensors[0].shape[0];
  // test
  std::cout << "1111111" << std::endl;
  tensors[0].PrintInfo();
  tensors[1].PrintInfo();
  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    if (multi_label_) {
      (*results)[bs].Reserve(tensors[0].shape[1] * (tensors[0].shape[2] - mask_nums_ - 5));
    } else {
      (*results)[bs].Reserve(tensors[0].shape[1]);
    }
    if (tensors[0].dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    const float* data = reinterpret_cast<const float*>(tensors[0].Data()) + bs * tensors[0].shape[1] * tensors[0].shape[2];
    // For Mask Proposals
    (*results)[bs].contain_masks = true;
    for (size_t i = 0; i < tensors[0].shape[1]; ++i) {
      int s = i * tensors[0].shape[2];
      float cls_conf = data[s + 4];
      float confidence = data[s + 4];
      std::vector<float> mask_proposal(data + s + tensors[0].shape[2] - mask_nums_, data + s + tensors[0].shape[2]);
      for (size_t k = 0; k < mask_proposal.size(); ++k) {
        mask_proposal[k] *= cls_conf;
      }
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

          // convert from [x, y, w, h] to [x1, y1, x2, y2]
          (*results)[bs].boxes.emplace_back(std::array<float, 4>{
              data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
              data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
              data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
          (*results)[bs].label_ids.push_back(label_id);
          (*results)[bs].scores.push_back(confidence);
          (*results)[bs].yolo_masks.push_back(mask_proposal);
        }
      } else {
        const float* max_class_score =
            std::max_element(data + s + 5, data + s + tensors[0].shape[2] - mask_nums_);
        confidence *= (*max_class_score);
        // filter boxes by conf_threshold
        if (confidence <= conf_threshold_) {
          continue;
        }
        int32_t label_id = std::distance(data + s + 5, max_class_score);
        // convert from [x, y, w, h] to [x1, y1, x2, y2]
        (*results)[bs].boxes.emplace_back(std::array<float, 4>{
            data[s] - data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] - data[s + 3] / 2.0f + label_id * max_wh_,
            data[s + 0] + data[s + 2] / 2.0f + label_id * max_wh_,
            data[s + 1] + data[s + 3] / 2.0f + label_id * max_wh_});
        (*results)[bs].label_ids.push_back(label_id);
        (*results)[bs].scores.push_back(confidence);
        (*results)[bs].yolo_masks.push_back(mask_proposal);
      }
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }
    std::cout << "before NMS" << std::endl;
    std::cout << (*results)[bs].yolo_masks.size() << std::endl;
    std::cout << (*results)[bs].boxes.size() << std::endl;
    std::cout << (*results)[bs].contain_masks << std::endl;
    utils::NMS(&((*results)[bs]), nms_threshold_);
    std::cout << "after NMS" << std::endl;
    std::cout << (*results)[bs].yolo_masks.size() << std::endl;
    std::cout << (*results)[bs].boxes.size() << std::endl;
    std::cout << (*results)[bs].contain_masks << std::endl;

    // deal with mask
    std::cout << "2222222222" << std::endl;
    (*results)[bs].masks.resize((*results)[bs].boxes.size());
    const float* data_mask = reinterpret_cast<const float*>(tensors[1].Data()) + bs * tensors[1].shape[1] * tensors[1].shape[2] * tensors[1].shape[3];
    cv::Mat mask_proto = cv::Mat(tensors[1].shape[1], tensors[1].shape[2] * tensors[1].shape[3], CV_32FC(1), const_cast<float*>(data_mask));
    std::cout << "333333333333" << std::endl;
    std::cout << mask_proto.size() << std::endl;
    std::cout << (*results)[bs].yolo_masks.size() << std::endl;
    std::cout << (*results)[bs].boxes.size() << std::endl;
    // vector to cv::Mat for Matmul
    cv::Mat mask_proposals;
    for (size_t i = 0; i < (*results)[bs].yolo_masks.size(); ++i) {
      std::string out;
      out = out + std::to_string((*results)[bs].boxes[i][0]) + "," +
          std::to_string((*results)[bs].boxes[i][1]) + ", " + std::to_string((*results)[bs].boxes[i][2]) +
          ", " + std::to_string((*results)[bs].boxes[i][3]);
      std::cout << out << std::endl;
      std::cout << "test_wjj111111:" << std::endl;
      for (size_t j = 0; j < 32; ++j) {
        std::cout << (*results)[bs].yolo_masks[i][j] << std::endl;
      }
      std::cout << cv::Mat((*results)[bs].yolo_masks[i]).size() << std::endl;
      mask_proposals.push_back(cv::Mat((*results)[bs].yolo_masks[i]).t());
    }
    std::cout << "444444444444" << std::endl;
    std::cout << mask_proposals.size() << std::endl;
    float* test_ptr_b = reinterpret_cast<float*>(mask_proposals.data);
    for (size_t m = 0; m < 20; ++m) {
      std::cout << test_ptr_b[m] << std::endl;
    }
    std::cout << "test_wjj_aaaaaaa" << std::endl;
    cv::Mat matmul_result = (mask_proposals * mask_proto).t();
    float* test_ptr_a = reinterpret_cast<float*>(matmul_result.data);
    for (size_t m = 0; m < 20; ++m) {
      std::cout << test_ptr_a[m] << std::endl;
    }
    std::cout << "5555555555555" << std::endl;
    std::cout << matmul_result.size() << std::endl;
    cv::Mat masks = matmul_result.reshape((*results)[bs].boxes.size(), {static_cast<int>(tensors[1].shape[2]), static_cast<int>(tensors[1].shape[3])});
    std::cout << "6666666666666" << std::endl;
    std::cout << masks.size() << std::endl;

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
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      int32_t label_id = ((*results)[bs].label_ids)[i];
      // clip box
      (*results)[bs].boxes[i][0] = (*results)[bs].boxes[i][0] - max_wh_ * label_id;
      (*results)[bs].boxes[i][1] = (*results)[bs].boxes[i][1] - max_wh_ * label_id;
      (*results)[bs].boxes[i][2] = (*results)[bs].boxes[i][2] - max_wh_ * label_id;
      (*results)[bs].boxes[i][3] = (*results)[bs].boxes[i][3] - max_wh_ * label_id;
      (*results)[bs].boxes[i][0] = std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][1] = std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][2] = std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][3] = std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w);
      (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h);
      (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w);
      (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h);
      // deal with mask
      cv::Mat dest, mask;
      // sigmoid
      cv::exp(-mask_channels[i], dest);
		  dest = 1.0 / (1.0 + dest);
		  // crop mask feature map
      int x1 = static_cast<int>(pad_w_mask);
      int y1 = static_cast<int>(pad_h_mask);
      int x2 = static_cast<int>(tensors[1].shape[3] - pad_w_mask);
      int y2 = static_cast<int>(tensors[1].shape[2] - pad_h_mask);
      float* test_ptr_c = reinterpret_cast<float*>(dest.data);
      for (size_t m = 0; m < 20; ++m) {
        std::cout << test_ptr_c[m] << std::endl;
      }
      std::cout << "7777777777" << std::endl;
      std::cout << dest.size() << std::endl;
      std::cout << x1 << ", " << y1 << ", " << x2 << ", " << y2 << std::endl;
		  cv::Rect roi(x1, y1, x2-x1, y2-y1);
      std::cout << roi << std::endl;
		  dest = dest(roi);
      float* test_ptr0 = reinterpret_cast<float*>(dest.data);
      for (size_t m = 0; m < 20; ++m) {
        std::cout << test_ptr0[m] << std::endl;
      }
      cv::resize(dest, mask, cv::Size(ipt_w, ipt_h), 0, 0, cv::INTER_LINEAR);
      std::cout << "88888888888" << std::endl;
      std::cout << mask.size() << std::endl;
		  // crop mask for source img
      int x1_src = static_cast<int>((*results)[bs].boxes[i][0]);
      int y1_src = static_cast<int>((*results)[bs].boxes[i][1]);
      int x2_src = static_cast<int>((*results)[bs].boxes[i][2]);
      int y2_src = static_cast<int>((*results)[bs].boxes[i][3]);
      cv::Rect roi_src(x1_src, y1_src, x2_src-x1_src, y2_src-y1_src);
      std::cout << roi_src << std::endl;
      mask = mask(roi_src);
      std::cout << "99999999999" << std::endl;
      std::cout << mask.size() << std::endl;
      float* test_ptr1 = reinterpret_cast<float*>(mask.data);
      for (size_t m = 0; m < 20; ++m) {
        std::cout << test_ptr1[m] << std::endl;
      }
		  mask = mask > mask_threshold_;
      // mask.convertTo(mask, CV_32SC1);
      int32_t* test_ptr2 = reinterpret_cast<int32_t*>(mask.ptr());
      for (size_t m = 0; m < 20; ++m) {
        std::cout << test_ptr2[m] << std::endl;
      }
      // save mask in DetectionResult
      int keep_mask_h = y2_src - y1_src;
      int keep_mask_w = x2_src - x1_src;
      int keep_mask_numel = keep_mask_h * keep_mask_w;
      (*results)[bs].masks[i].Resize(keep_mask_numel);
      (*results)[bs].masks[i].shape = {keep_mask_h, keep_mask_w};
      uint8_t* keep_mask_ptr =
          reinterpret_cast<uint8_t*>((*results)[bs].masks[i].Data());
      std::memcpy(keep_mask_ptr, reinterpret_cast<uint8_t*>(mask.ptr()), keep_mask_numel * sizeof(uint8_t));
    }
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
