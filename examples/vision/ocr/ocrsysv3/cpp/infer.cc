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

int main() {
  //模型路径准备
  std::string det_model_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_det_infer/inference.pdmodel";
  std::string det_params_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_det_infer/inference.pdiparams";
  std::string cls_model_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel";
  std::string cls_params_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams";
  std::string rec_model_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_rec_infer/inference.pdmodel";
  std::string rec_params_file =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_rec_infer/inference.pdiparams";

  //识别器label准备
  std::string rec_label_path =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_rec_infer/ppocr_keys_v1.txt";

  //输入图片，可视化图片准备, 1张
  // std::string img_path =
  // "/xieyunyao/project/FastDeploy/examples/resources/images/word_1.jpg";
  std::string img_path =
      "/xieyunyao/project/FastDeploy/examples/resources/images/11.jpg";
  std::string vis_path =
      "/xieyunyao/project/FastDeploy/examples/resources/outputs/vis.jpeg";

  //准备图片list
  std::vector<cv::String> cv_all_img_names;
  cv::glob(img_path, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  auto option1 = fastdeploy::RuntimeOption();
  option1.UseCpu();
  option1.UseOrtBackend();
  auto option2 = fastdeploy::RuntimeOption();
  option2.UseCpu();
  option2.UseOrtBackend();
  auto option3 = fastdeploy::RuntimeOption();
  option3.UseGpu(5);
  option3.UseOrtBackend();

  // option3.UseTrtBackend();
  // option3.SetTrtInputShape("x", {1, 3, 48, 10}, {1, 3, 48, 320},
  //                         {1, 3, 48, 2000});
  // option3.SetTrtInputShape("lstm_0.tmp_0", {10, 1, 96}, {25, 1, 96},
  //                         {1000, 1, 96});

  auto det_model = fastdeploy::vision::ppocr::DBDetector(
      det_model_file, det_params_file, option1);
  auto cls_model = fastdeploy::vision::ppocr::Classifier(
      cls_model_file, cls_params_file, option2);
  auto rec_model = fastdeploy::vision::ppocr::Recognizer(
      rec_label_path, rec_model_file, rec_params_file, option3);

  // auto ocrv3_app =
  // fastdeploy::application::ocrsystem::PPOCRSystemv3(&det_model,&cls_model,&rec_model);
  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  std::vector<cv::Mat> img_list;
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);

    if (!srcimg.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      std::exit(1);
    }
    img_list.push_back(srcimg);
  }

  //预测
  std::vector<std::vector<fastdeploy::vision::OCRResult>> ocr_results =
      ocrv3_app.Predict(img_list);

  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    std::cout << "Image name is: " << cv_all_img_names[i] << std::endl;

    //输出预测信息
    for (int j = 0; j < ocr_results[i].size(); ++j) {
      std::cout << ocr_results[i][j].Str() << std::endl;
    }

    //可视化
    cv::Mat im_bak = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    auto vis_img =
        fastdeploy::vision::Visualize::VisOcr(im_bak, ocr_results[i]);
    cv::imwrite(vis_path, vis_img);
    std::cout << "Ocr Predict Done! Visualized image is saved: " << vis_path
              << std::endl;
  }

  return 0;
}