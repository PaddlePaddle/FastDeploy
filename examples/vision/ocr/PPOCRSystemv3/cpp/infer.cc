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

void CpuInfer(const std::string& det_model_dir,
              const std::string& cls_model_dir,
              const std::string& rec_model_dir,
              const std::string& rec_label_file,
              const std::string& image_file) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";
  auto rec_label = rec_label_file;

  std::string vis_path = "./vis.jpg";

  //准备runtime
  auto det_option = fastdeploy::RuntimeOption();
  det_option.UseCpu();
  auto cls_option = fastdeploy::RuntimeOption();  //只可用PaddleBackend
  cls_option.UseCpu();
  auto rec_option = fastdeploy::RuntimeOption();
  rec_option.UseCpu();

  //准备模型
  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  if (!det_model.Initialized()) {
    std::cerr << "Failed to initialize det_model." << std::endl;
    return;
  }
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  if (!cls_model.Initialized()) {
    std::cerr << "Failed to initialize cls_model." << std::endl;
    return;
  }
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label, rec_option);
  if (!rec_model.Initialized()) {
    std::cerr << "Failed to initialize rec_model." << std::endl;
    return;
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  //准备输入
  std::vector<cv::String> cv_all_img_names;
  cv::glob(image_file, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

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

  //开始预测
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
}

void GpuInfer(const std::string& det_model_dir,
              const std::string& cls_model_dir,
              const std::string& rec_model_dir,
              const std::string& rec_label_file,
              const std::string& image_file) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";
  auto rec_label = rec_label_file;

  std::string vis_path = "./vis.jpg";

  //准备runtime
  auto det_option = fastdeploy::RuntimeOption();
  det_option.UseGpu();
  auto cls_option = fastdeploy::RuntimeOption();
  cls_option.UseGpu();
  auto rec_option = fastdeploy::RuntimeOption();
  rec_option.UseGpu();

  //准备模型
  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  if (!det_model.Initialized()) {
    std::cerr << "Failed to initialize det_model." << std::endl;
    return;
  }
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  if (!cls_model.Initialized()) {
    std::cerr << "Failed to initialize cls_model." << std::endl;
    return;
  }
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label, rec_option);
  if (!rec_model.Initialized()) {
    std::cerr << "Failed to initialize rec_model." << std::endl;
    return;
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  //准备输入
  std::vector<cv::String> cv_all_img_names;
  cv::glob(image_file, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

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

  //开始预测
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
}

void TrtInfer(const std::string& det_model_dir,
              const std::string& cls_model_dir,
              const std::string& rec_model_dir,
              const std::string& rec_label_file,
              const std::string& image_file) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";
  auto rec_label = rec_label_file;

  std::string vis_path = "./vis.jpg";

  //准备runtime
  auto det_option = fastdeploy::RuntimeOption();
  det_option.UseGpu();
  det_option.UseTrtBackend();
  det_option.SetTrtInputShape("x", {1, 3, 50, 50}, {1, 3, 640, 640},
                              {1, 3, 960, 960});

  auto cls_option = fastdeploy::RuntimeOption();
  cls_option.UseGpu();
  cls_option.UseTrtBackend();
  cls_option.SetTrtInputShape("x", {1, 3, 32, 100});

  auto rec_option = fastdeploy::RuntimeOption();
  rec_option.UseGpu();
  rec_option.UseTrtBackend();
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {1, 3, 48, 320},
                              {1, 3, 48, 2000});

  //准备模型
  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  if (!det_model.Initialized()) {
    std::cerr << "Failed to initialize det_model." << std::endl;
    return;
  }
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  if (!cls_model.Initialized()) {
    std::cerr << "Failed to initialize cls_model." << std::endl;
    return;
  }
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label, rec_option);
  if (!rec_model.Initialized()) {
    std::cerr << "Failed to initialize rec_model." << std::endl;
    return;
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  //准备输入
  std::vector<cv::String> cv_all_img_names;
  cv::glob(image_file, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

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

  //开始预测
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
}

int main(int argc, char* argv[]) {
  if (argc < 7) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[6]) == 0) {
    CpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } else if (std::atoi(argv[6]) == 1) {
    GpuInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  } else if (std::atoi(argv[6]) == 2) {
    TrtInfer(argv[1], argv[2], argv[3], argv[4], argv[5]);
  }
  return 0;
}