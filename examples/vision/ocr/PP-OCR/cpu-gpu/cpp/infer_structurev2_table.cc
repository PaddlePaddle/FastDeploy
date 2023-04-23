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

void InitAndInfer(const std::string &table_model_dir,
                  const std::string &image_file,
                  const std::string &table_char_dict_path,
                  const fastdeploy::RuntimeOption &option) {
  auto table_model_file = table_model_dir + sep + "inference.pdmodel";
  auto table_params_file = table_model_dir + sep + "inference.pdiparams";
  auto table_option = option;

  auto table_model = fastdeploy::vision::ocr::StructureV2Table(
      table_model_file, table_params_file, table_char_dict_path, table_option);
  assert(table_model.Initialized());

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!table_model.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cout << "Usage: infer_demo path/to/table_model path/to/image  "
                 "path/to/table_dict_path"
                 "run_option, "
                 "e.g ./infer_structurev2_table ch_ppocr_mobile_v2.0_cls_infer "
                 "table.jpg table_structure_dict.txt 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu;."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[4]);

  if (flag == 0) {
    option.UseCpu();
  } else if (flag == 1) {
    option.UseGpu();
  }

  std::string table_model_dir = argv[1];
  std::string test_image = argv[2];
  std::string table_char_dict_path = argv[3];
  InitAndInfer(table_model_dir, test_image, table_char_dict_path, option);
  return 0;
}
