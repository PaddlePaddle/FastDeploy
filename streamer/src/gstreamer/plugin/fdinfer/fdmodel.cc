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

#include "fdmodel.h"

#include <iostream>

namespace fastdeploy {
namespace streamer {

void* CreateModel(const std::string& model_name, RuntimeOption& option,
                  const std::string& model_file, const std::string& params_file,
                  const std::string& config_file) {
  if (model_name == "PPYOLOE") {
    auto model = new fastdeploy::vision::detection::PPYOLOE(
        model_file, params_file, config_file, option);
    return reinterpret_cast<void*>(model);
  } else {
    FDASSERT(false, "Unsupported model: %s", model_name.c_str());
  }
  return nullptr;
}

bool ModelPredict(const std::string& model_name, void* model, GstBuffer* inbuf,
                  int width, int height,
                  fastdeploy::vision::DetectionResult& res) {
  GstMapInfo in_map_info;
  memset(&in_map_info, 0, sizeof(in_map_info));
  if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READ)) {
    return false;
  }
  cv::Mat im(height, width, CV_8UC3, in_map_info.data);
  bool ret = false;
  if (model_name == "PPYOLOE") {
    auto fd_model =
        reinterpret_cast<fastdeploy::vision::detection::PPYOLOE*>(model);
    ret = fd_model->Predict(im, &res);
  }
  if (!ret) {
    std::cerr << "Failed to predict." << std::endl;
    return false;
  }
  std::cout << "num of bbox: " << res.boxes.size() << std::endl;
  for (size_t i = 0; i < res.boxes.size(); ++i) {
    if (res.scores[i] < 0.5) {
      continue;
    }
    int x1 = static_cast<int>(round(res.boxes[i][0]));
    int y1 = static_cast<int>(round(res.boxes[i][1]));
    int x2 = static_cast<int>(round(res.boxes[i][2]));
    int y2 = static_cast<int>(round(res.boxes[i][3]));
    cv::Scalar rect_color = cv::Scalar(0, 255, 0);
    cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
    cv::rectangle(im, rect, rect_color, 1);
    std::string out = "";
    out = out + std::to_string(res.boxes[i][0]) + "," +
          std::to_string(res.boxes[i][1]) + ", " +
          std::to_string(res.boxes[i][2]) + ", " +
          std::to_string(res.boxes[i][3]) + ", " +
          std::to_string(res.scores[i]) + ", " +
          std::to_string(res.label_ids[i]);
    std::cout << out << std::endl;
  }

  fastdeploy::vision::VisDetection(im, res, 0.5);

  gst_buffer_unmap(inbuf, &in_map_info);

  return true;
}

}  // namespace streamer
}  // namespace fastdeploy
