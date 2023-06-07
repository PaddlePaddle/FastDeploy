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

#include "flags.h"
#include "macros.h"
#include "option.h"

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;

DEFINE_string(table_char_dict_path, "",
              "Path of table character dict of PPOCR.");
DEFINE_string(trt_shape, "1,3,48,10:4,3,48,320:8,3,48,2304",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,48,10:4,3,48,320:8,3,48,2304");

int main(int argc, char *argv[]) {
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
  auto model_format = fastdeploy::ModelFormat::PADDLE;
  if (!UpdateModelResourceName(&model_name, &params_name, &config_name,
                               &model_format, config_info, false)) {
    return -1;
  }
  auto model_file = FLAGS_model + sep + model_name;
  auto params_file = FLAGS_model + sep + params_name;
  if (config_info["backend"] == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (config_info["backend"] == "paddle_trt" ||
      config_info["backend"] == "trt") {
    std::vector<std::vector<int32_t>> trt_shapes =
        benchmark::ResultManager::GetInputShapes(FLAGS_trt_shape);
    option.trt_option.SetShape("x", trt_shapes[0], trt_shapes[1],
                               trt_shapes[2]);
  }

  auto model_ppocr_table = vision::ocr::StructureV2Table(
      model_file, params_file, FLAGS_table_char_dict_path, option,
      model_format);
  fastdeploy::vision::OCRResult result;

  if (config_info["precision_compare"] == "true") {
    std::string expect_structure_html =
        "<html><body><table><thead><tr><td></td><td></td><td></td><td></"
        "td><td></td></tr></thead><tbody><tr><td></td><td></td><td></td><td></"
        "td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></"
        "tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></"
        "td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></"
        "td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></"
        "td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></"
        "td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></"
        "tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></"
        "td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></"
        "td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></"
        "td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></"
        "td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></"
        "tr><tr><td></td><td></td><td></td><td></td><td></td></tr></tbody></"
        "table></body></html>";
    std::vector<int> expect_box_coord{
        41,  4,   97,  18,  161, 4,   173, 18,  216, 4,   225, 17,  272, 4,
        283, 17,  321, 4,   348, 18,  33,  20,  106, 38,  150, 22,  180, 38,
        202, 22,  235, 38,  262, 21,  293, 38,  326, 23,  343, 37,  27,  38,
        109, 56,  150, 39,  179, 56,  204, 39,  236, 56,  263, 39,  292, 55,
        329, 40,  343, 54,  22,  57,  118, 74,  152, 58,  176, 74,  204, 58,
        236, 75,  262, 58,  291, 74,  326, 58,  344, 74,  27,  75,  119, 92,
        150, 75,  177, 92,  204, 75,  235, 92,  260, 75,  292, 92,  326, 75,
        346, 92,  44,  92,  102, 110, 150, 92,  177, 110, 205, 92,  236, 110,
        262, 92,  290, 110, 329, 93,  339, 110, 41,  109, 102, 128, 151, 110,
        175, 128, 205, 110, 236, 128, 262, 110, 291, 127, 329, 110, 338, 127,
        42,  128, 102, 146, 149, 128, 177, 146, 205, 128, 237, 146, 262, 128,
        291, 146, 329, 128, 339, 145, 31,  145, 110, 163, 150, 145, 178, 163,
        206, 145, 237, 164, 262, 145, 292, 163, 324, 145, 342, 162, 40,  162,
        108, 180, 154, 162, 175, 180, 209, 162, 231, 180, 266, 162, 286, 180,
        325, 162, 341, 179, 38,  180, 105, 197, 152, 180, 177, 197, 207, 180,
        236, 197, 262, 180, 291, 197, 329, 181, 339, 196, 42,  196, 102, 214,
        151, 197, 179, 214, 205, 197, 236, 214, 263, 197, 291, 214, 320, 197,
        349, 214, 46,  215, 100, 233, 149, 216, 179, 233, 204, 216, 238, 233,
        262, 216, 291, 233, 321, 216, 345, 232, 42,  233, 104, 251, 147, 234,
        179, 251, 203, 233, 237, 251, 260, 233, 294, 251, 326, 234, 341, 250,
        19,  251, 120, 269, 148, 253, 180, 270, 202, 252, 240, 270, 259, 252,
        294, 270, 324, 252, 347, 268, 16,  270, 123, 286, 146, 270, 182, 287,
        200, 270, 238, 287, 256, 270, 294, 286, 319, 270, 353, 286};

    // Run once at least
    if (!model_ppocr_table.Predict(im, &result)) {
      std::cerr << "Failed to predict." << std::endl;
      return -1;
    }

    // 1. Test result diff
    std::cout << "=============== Test Table Result diff =================\n";
    // Calculate diff between two results.
    std::string result_table_structure;
    for (auto &structure : result.table_structure) {
      result_table_structure += structure;
    }
    if (expect_structure_html == result_table_structure) {
      std::cout << "PPOCR Table structure has no diff" << std::endl;
    } else {
      std::cout << "PPOCR Table structure has diff" << std::endl;
      std::cout << "expected: " << expect_structure_html << std::endl;
      std::cout << "result: " << result_table_structure << std::endl;
    }

    std::vector<int> table_box_coord;
    for (auto &box : result.table_boxes) {
      // x1 y1 x2 y1 x2 y2 x1 y2 => x1 y1 x2 y2
      table_box_coord.push_back(box[0]);
      table_box_coord.push_back(box[1]);
      table_box_coord.push_back(box[2]);
      table_box_coord.push_back(box[5]);
    }

    if (expect_box_coord.size() == table_box_coord.size()) {
      std::cout << "table boxes num matched with expected: "
                << table_box_coord.size() << std::endl;
      int max_diff = 0;
      int total_diff = 0;
      for (int i = 0; i < table_box_coord.size(); i++) {
        int diff = std::abs(table_box_coord[i] - expect_box_coord[i]);
        if (diff > max_diff) {
          max_diff = diff;
        }
        total_diff += diff;
      }
      std::cout << "box coords, max_diff: " << max_diff << ", "
                << ", total diff: " << total_diff << ", average diff: "
                << total_diff / float(table_box_coord.size()) << std::endl;
    } else {
      std::cout << "boxes num has diff, expect box num: "
                << expect_box_coord.size() / 4
                << ", result box num:" << table_box_coord.size() / 4
                << std::endl;
    }
  }

  BENCHMARK_MODEL(model_ppocr_table, model_ppocr_table.Predict(im, &result));
#endif
  return 0;
}