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

#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string &layout_model_dir,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto layout_model_file = layout_model_dir + sep + "model.pdmodel";
  auto layout_params_file = layout_model_dir + sep + "model.pdiparams";

  auto layout_model = fastdeploy::vision::ocr::StructureV2Layout(
      layout_model_file, layout_params_file, option);

  if (!layout_model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  // 5 for publaynet, 10 for cdla
  layout_model.GetPostprocessor().SetNumClass(5);
  fastdeploy::vision::DetectionResult res;
  if (!layout_model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  std::vector<std::string> labels = {"text", "title", "list", "table",
                                     "figure"};
  if (layout_model.GetPostprocessor().GetNumClass() == 10) {
    labels = {"text",      "title",         "figure", "figure_caption",
              "table",     "table_caption", "header", "footer",
              "reference", "equation"};
  }
  auto vis_im = fastdeploy::vision::VisDetection(im, res, labels, 0.3, 2, .5f,
                                                 {255, 0, 0}, 2);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout
        << "Usage: infer_demo path/to/layout_model path/to/image  "
           "run_option, "
           "e.g ./infer_structurev2_layout picodet_lcnet_x1_0_fgd_layout_infer "
           "layout.png 0"
        << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu;."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[3]);

  if (flag == 0) {
    option.UseCpu();
  } else if (flag == 1) {
    option.UseGpu();
  }

  std::string layout_model_dir = argv[1];
  std::string image_file = argv[2];
  InitAndInfer(layout_model_dir, image_file, option);
  return 0;
}
