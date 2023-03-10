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

DEFINE_string(trt_shape, "1,3,512,512:1,3,512,512:1,3,512,512",
              "Set min/opt/max shape for trt/paddle_trt backend."
              "eg:--trt_shape 1,3,512,512:1,3,512,512:1,3,512,512");
DEFINE_bool(quant, false, "Whether to use quantize model");

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
  auto model_file = FLAGS_model + sep + "model.pdmodel";
  auto params_file = FLAGS_model + sep + "model.pdiparams";
  auto config_file = FLAGS_model + sep + "deploy.yaml";
  auto model_format = fastdeploy::ModelFormat::PADDLE;
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
  if (config_info["backend"] == "mnn") {
    model_file = FLAGS_model + sep + "model.mnn";
    if (FLAGS_quant) {
      model_file = FLAGS_model + sep + "model_quant.mnn";
    }
    params_file = "";
    model_format = fastdeploy::ModelFormat::MNN_MODEL;
  } else if (config_info["backend"] == "tnn") {
    model_file = FLAGS_model + sep + "model.opt.tnnmodel";
    params_file = FLAGS_model + sep + "model.opt.tnnproto";
    model_format = fastdeploy::ModelFormat::TNN_MODEL;
  } else if (config_info["backend"] == "ncnn") {
    model_file = FLAGS_model + sep + "model.opt.bin";
    params_file = FLAGS_model + sep + "model.opt.param";
    model_format = fastdeploy::ModelFormat::NCNN_MODEL;
  }
  auto model_ppmatting = vision::matting::PPMatting(
      model_file, params_file, config_file, option, model_format);
  vision::MattingResult res;
  if (config_info["precision_compare"] == "true") {
    std::cout << "precision_compare for PPMatting is not support now!"
              << std::endl;
  }
  BENCHMARK_MODEL(model_ppmatting, model_ppmatting.Predict(&im, &res))
  auto vis_im = vision::VisMatting(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif
  return 0;
}