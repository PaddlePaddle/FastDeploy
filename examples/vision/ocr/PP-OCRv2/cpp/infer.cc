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

void InitAndInfer(const std::string& det_model_dir, const std::string& cls_model_dir, const std::string& rec_model_dir, const std::string& rec_label_file, const std::string& image_file, const fastdeploy::RuntimeOption& option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_model = fastdeploy::vision::ocr::DBDetector(det_model_file, det_params_file, option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(cls_model_file, cls_params_file, option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(rec_model_file, rec_params_file, rec_label_file, option);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());
  
  // The classification model is optional, so the PP-OCR can also be connected in series as follows
  // auto ppocr_v2 = fastdeploy::pipeline::PPOCRv2(&det_model, &rec_model);
  auto ppocr_v2 = fastdeploy::pipeline::PPOCRv2(&det_model, &cls_model, &rec_model);

  if(!ppocr_v2.Initialized()){
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();
  
  fastdeploy::vision::OCRResult result;
  if (!ppocr_v2.Predict(&im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv2_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[6]);

  if (flag == 0) {
    option.UseCpu(); 
  } else if (flag == 1) {
    option.UseGpu();
  } else if (flag == 2) {
    option.UseGpu();
    option.UseTrtBackend();
  }

  std::string det_model_dir = argv[1];
  std::string cls_model_dir = argv[2];
  std::string rec_model_dir = argv[3];
  std::string rec_label_file = argv[4];
  std::string test_image = argv[5];
  InitAndInfer(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file, test_image, option);
  return 0;
}
