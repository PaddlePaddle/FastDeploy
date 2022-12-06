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

#include "fastdeploy/vision/facedet/yolov7-face/postprocessor.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace facedet {

Yolov7FacePostprocessor::Yolov7FacePostprocessor() {
  conf_threshold_ = 0.7;
  nms_threshold_ = 0.5;
  max_wh_ = 7680.0;
}

bool Yolov7FacePostprocessor::Run(const std::vector<FDTensor>& infer_result, std::vector<FaceDetectionResult>* results,
                              const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {
  int batch = infer_result[0].shape[0];

  std::cout <<"batch size:" << batch << std::endl;
 
  results->resize(batch);

  for (size_t bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    (*results)[bs].Reserve(infer_result[0].shape[1]);
    if (infer_result[0].dtype != FDDataType::FP32) {
      FDERROR << "Only support post process with float32 data." << std::endl;
      return false;
    }
    const float* data = reinterpret_cast<const float*>(infer_result[0].Data()) + bs * infer_result[0].shape[1] * infer_result[0].shape[2];
    for (size_t i = 0; i < infer_result[0].shape[1]; ++i) {
      int s = i * infer_result[0].shape[2];
      float confidence = data[s + 4];
      const float* reg_cls_ptr = data + s;
      const float* class_score = data + s + 5;
      confidence  *= (*class_score);
      // filter boxes by conf_threshold
      if (confidence <= conf_threshold_) {
        continue;
      }
      float x = reg_cls_ptr[0];
      float y = reg_cls_ptr[1];
      float w = reg_cls_ptr[2];
      float h = reg_cls_ptr[3];

      // convert from [x, y, w, h] to [x1, y1, x2, y2]
      (*results)[bs].boxes.emplace_back(std::array<float, 4>{
          (x - w / 2.f), (y - h / 2.f), (x + w / 2.f), (y + h / 2.f)});
      (*results)[bs].scores.push_back(confidence);
    }

    if ((*results)[bs].boxes.size() == 0) {
      return true;
    }
  
    //std::cout << "Before: "<< (*results)[bs].Str() << std::endl;
    utils::NMS(&((*results)[bs]), nms_threshold_);
    //std::cout << "After NMS: "<< (*results)[bs].Str() << std::endl;

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
    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      float pad_h = (out_h - ipt_h * scale) / 2;
      float pad_w = (out_w - ipt_w * scale) / 2;
      // clip box
      (*results)[bs].boxes[i][0] = std::max(((*results)[bs].boxes[i][0] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][1] = std::max(((*results)[bs].boxes[i][1] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][2] = std::max(((*results)[bs].boxes[i][2] - pad_w) / scale, 0.0f);
      (*results)[bs].boxes[i][3] = std::max(((*results)[bs].boxes[i][3] - pad_h) / scale, 0.0f);
      (*results)[bs].boxes[i][0] = std::min((*results)[bs].boxes[i][0], ipt_w - 1.0f);
      (*results)[bs].boxes[i][1] = std::min((*results)[bs].boxes[i][1], ipt_h - 1.0f);
      (*results)[bs].boxes[i][2] = std::min((*results)[bs].boxes[i][2], ipt_w - 1.0f);
      (*results)[bs].boxes[i][3] = std::min((*results)[bs].boxes[i][3], ipt_h - 1.0f);
    }
  }
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
