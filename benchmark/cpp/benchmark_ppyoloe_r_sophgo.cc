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

#include <fstream>

#include "flags.h"
#include "macros.h"
#include "option.h"

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;

DEFINE_bool(no_nms, false, "Whether the model contains nms.");

static inline void StrSplitToFloat(std::vector<float>& res,
                                   const std::string& s, const std::string& c) {
  std::string::size_type pos1, pos2;
  pos2 = s.find(c);
  pos1 = 0;
  while (std::string::npos != pos2) {
    res.emplace_back(std::stof(s.substr(pos1, pos2 - pos1)));
    pos1 = pos2 + strlen(c.c_str());
    pos2 = s.find(c, pos1);
  }
  res.emplace_back(std::stof(s.substr(pos1)));
}

void showDiffStats(const std::vector<float>& data, const std::string& title) {
  float max{0.0};
  float min{0.0};
  float average{0.0};

  if (data.size() == 0) {
    std::cout << title << ", max: " << max << ", min: " << min
              << ", mean: " << average << std::endl;
    return;
  }
  double sum = accumulate(begin(data), end(data), 0.0);
  double mean = sum / data.size();
  max = *max_element(data.begin(), data.end());
  min = *min_element(data.begin(), data.end());
  average = mean;
  std::cout << title << " diff, max: " << max << ", min: " << min
            << ", mean: " << average << std::endl;
}

void sortBoxes(vision::DetectionResult* result, std::vector<int>* indices) {
  if (result->rotated_boxes.empty()) {
    return;
  }

  indices->clear();
  indices->resize(result->rotated_boxes.size());
  for (int i = 0; i < result->rotated_boxes.size(); ++i) {
    indices->at(i) = i;
  }

  auto& rotated_boxes = result->rotated_boxes;
  std::sort(indices->begin(), indices->end(), [&rotated_boxes](int a, int b) {
    if (rotated_boxes[a][0] == rotated_boxes[b][0]) {
      return rotated_boxes[a][1] > rotated_boxes[b][1];
    }
    return rotated_boxes[a][0] > rotated_boxes[b][0];
  });
}

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  std::string model_name, params_name, config_name;
  auto model_format = fastdeploy::ModelFormat::SOPHGO;
  auto model_file =
      FLAGS_model + sep + "ppyoloe_r_crn_s_3x_dota_1684x_f32.bmodel";
  auto params_file = "";
  auto config_file = FLAGS_model + sep + "infer_cfg.yml";

  auto model_ppyoloe_r = vision::detection::PPYOLOE_R(
      model_file, params_file, config_file, option, model_format);

  vision::DetectionResult res;
  if (config_info["precision_compare"] == "true") {
    // Run once at least
    model_ppyoloe_r.Predict(im, &res);
    // 1. Test result diff
    std::cout << "=============== Test result diff =================\n";

    // result from PaddleDetection, cls_id, conf, boxes(8)
    std::string det_result_path = "ppyoloe_r_result.txt";
    std::ifstream ifs(det_result_path.c_str());
    std::string expect_res((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
    ifs.close();
    std::vector<float> expect_data;
    StrSplitToFloat(expect_data, expect_res, ",");

    std::vector<float> label_diff;
    std::vector<float> score_diff;
    std::vector<float> boxes_diff;
    int num = int(res.label_ids.size());
    if (expect_data.size() == num) {
      std::cout << "boxes num is the same: " << expect_data.size() << std::endl;
    }

    vision::DetectionResult expect_result;
    for (int i = 0; i < num; i++) {
      expect_result.label_ids.push_back(
          static_cast<int32_t>(expect_data[10 * i]));
      expect_result.scores.push_back(expect_data[10 * i + 1]);
      std::array<float, 8> box;
      for (int j = 0; j < 8; j++) {
        box[j] = expect_data[10 * i + 2 + j];
      }
      expect_result.rotated_boxes.push_back(box);
    }
    std::vector<int> exp_indices;
    sortBoxes(&expect_result, &exp_indices);

    std::vector<int> res_indices;
    sortBoxes(&res, &res_indices);

    for (int i = 0; i < num; i++) {
      int res_idx = res_indices[i];
      int exp_idx = exp_indices[i];
      label_diff.push_back(
          std::abs(expect_result.label_ids[exp_idx] - res.label_ids[res_idx]));
      score_diff.push_back(
          std::abs(expect_result.scores[exp_idx] - res.scores[res_idx]));
      for (int j = 0; j < 8; j++) {
        boxes_diff.push_back(std::abs(expect_result.rotated_boxes[exp_idx][j] -
                                      res.rotated_boxes[res_idx][j]));
      }
    }
    showDiffStats(label_diff, "labels");
    showDiffStats(score_diff, "scores");
    showDiffStats(boxes_diff, "boxes");
  }

#endif
  return 0;
}