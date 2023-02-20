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

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  auto im_rec = cv::imread(FLAGS_image_rec);
  // Detection Model
  auto det_model_file =
      FLAGS_model + sep + FLAGS_det_model + sep + "inference.pdmodel";
  auto det_params_file =
      FLAGS_model + sep + FLAGS_det_model + sep + "inference.pdiparams";
  // Classification Model
  auto cls_model_file =
      FLAGS_model + sep + FLAGS_cls_model + sep + "inference.pdmodel";
  auto cls_params_file =
      FLAGS_model + sep + FLAGS_cls_model + sep + "inference.pdiparams";
  // Recognition Model
  auto rec_model_file =
      FLAGS_model + sep + FLAGS_rec_model + sep + "inference.pdmodel";
  auto rec_params_file =
      FLAGS_model + sep + FLAGS_rec_model + sep + "inference.pdiparams";
  auto rec_label_file = FLAGS_rec_label_file;
  if (FLAGS_backend == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;
  if (FLAGS_backend == "paddle_trt" || FLAGS_backend == "trt") {
    det_option.trt_option.SetShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                                   {1, 3, 960, 960});
    cls_option.trt_option.SetShape("x", {1, 3, 48, 10}, {4, 3, 48, 320},
                                   {8, 3, 48, 1024});
    rec_option.trt_option.SetShape("x", {1, 3, 48, 10}, {4, 3, 48, 320},
                                   {8, 3, 48, 2304});
  }
  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);
  // Only for runtime
  if (FLAGS_profile_mode == "runtime") {
    std::vector<std::array<int, 8>> boxes_result;
    std::cout << "====Detection model====" << std::endl;
    BENCHMARK_MODEL(det_model, det_model.Predict(im, &boxes_result));
    int32_t cls_label;
    float cls_score;
    std::cout << "====Classification model====" << std::endl;
    BENCHMARK_MODEL(cls_model,
                    cls_model.Predict(im_rec, &cls_label, &cls_score));
    std::string text;
    float rec_score;
    std::cout << "====Recognization model====" << std::endl;
    BENCHMARK_MODEL(rec_model, rec_model.Predict(im_rec, &text, &rec_score));
  }
  auto model_ppocrv3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);
  fastdeploy::vision::OCRResult res;
  if (FLAGS_profile_mode == "end2end") {
    BENCHMARK_MODEL(model_ppocrv3, model_ppocrv3.Predict(im, &res))
  }
  auto vis_im = fastdeploy::vision::VisOcr(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif
  return 0;
}