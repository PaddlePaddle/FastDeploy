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
  std::string det_model_dir =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_det_infer";
  std::string cls_model_dir =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_ppocr_mobile_v2.0_cls_infer";
  std::string rec_model_dir =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_rec_infer";

  //识别器label准备
  std::string rec_label_path =
      "/xieyunyao/project/FastDeploy/examples/resources/models/"
      "ch_PP-OCRv3_rec_infer/ppocr_keys_v1.txt";

  //输入图片，可视化图片准备, 1张
  std::string img_path =
      "/xieyunyao/project/FastDeploy/examples/resources/images/12.jpg";
  std::string vis_path =
      "/xieyunyao/project/FastDeploy/examples/resources/outputs/vis.jpeg";

  //准备图片list
  std::vector<cv::String> cv_all_img_names;
  cv::glob(img_path, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  //准备Runtime
  auto one_runtime = fastdeploy::RuntimeOption();
  //初始化OCR系统
  bool use_det = true;
  bool use_cls = true;
  bool use_rec = true;
  // auto ppocrsys = fastdeploy::vision::ppocr::PPocrsys(
  //     use_det, use_cls, use_rec, rec_label_path, det_model_file,
  //     cls_model_file,
  //     rec_model_file, det_params_file, cls_params_file, rec_params_file,
  //     det_option, cls_option, rec_option);

  fastdeploy::vision::ppocr::PPocrsys ppocrsys(
      use_det, use_cls, use_rec, rec_label_path, det_model_dir, cls_model_dir,
      rec_model_dir, one_runtime);

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
      ppocrsys.ocrsys(img_list, use_det, use_cls, use_rec);

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