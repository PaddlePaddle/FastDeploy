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

  fastdeploy::vision::ocr::DBDetector det_model;
  fastdeploy::vision::ocr::Classifier cls_model;
  fastdeploy::vision::ocr::Recognizer rec_model;

  if (!det_model_dir.empty()) {
    auto det_option = fastdeploy::RuntimeOption();
    det_option.UseCpu();
    det_model = fastdeploy::vision::ocr::DBDetector(
        det_model_file, det_params_file, det_option);

    if (!det_model.Initialized()) {
      std::cerr << "Failed to initialize det_model." << std::endl;
      return;
    }
  }

  if (!cls_model_dir.empty()) {
    auto cls_option = fastdeploy::RuntimeOption();
    cls_option.UseCpu();
    cls_model = fastdeploy::vision::ocr::Classifier(
        cls_model_file, cls_params_file, cls_option);

    if (!cls_model.Initialized()) {
      std::cerr << "Failed to initialize cls_model." << std::endl;
      return;
    }
  }

  if (!rec_model_dir.empty()) {
    auto rec_option = fastdeploy::RuntimeOption();
    rec_option.UseCpu();
    rec_model = fastdeploy::vision::ocr::Recognizer(
        rec_model_file, rec_params_file, rec_label, rec_option);

    if (!rec_model.Initialized()) {
      std::cerr << "Failed to initialize rec_model." << std::endl;
      return;
    }
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult res;
  //开始预测
  if (!ocrv3_app.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  //输出预测信息
  std::cout << res.Str() << std::endl;

  //可视化
  auto vis_img = fastdeploy::vision::Visualize::VisOcr(im_bak, res);

  cv::imwrite("vis_result.jpg", vis_img);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
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

  fastdeploy::vision::ocr::DBDetector det_model;
  fastdeploy::vision::ocr::Classifier cls_model;
  fastdeploy::vision::ocr::Recognizer rec_model;

  //准备模型
  if (!det_model_dir.empty()) {
    auto det_option = fastdeploy::RuntimeOption();
    det_option.UseGpu();
    det_model = fastdeploy::vision::ocr::DBDetector(
        det_model_file, det_params_file, det_option);

    if (!det_model.Initialized()) {
      std::cerr << "Failed to initialize det_model." << std::endl;
      return;
    }
  }

  if (!cls_model_dir.empty()) {
    auto cls_option = fastdeploy::RuntimeOption();
    cls_option.UseGpu();
    cls_model = fastdeploy::vision::ocr::Classifier(
        cls_model_file, cls_params_file, cls_option);

    if (!cls_model.Initialized()) {
      std::cerr << "Failed to initialize cls_model." << std::endl;
      return;
    }
  }

  if (!rec_model_dir.empty()) {
    auto rec_option = fastdeploy::RuntimeOption();
    rec_option.UseGpu();
    rec_model = fastdeploy::vision::ocr::Recognizer(
        rec_model_file, rec_params_file, rec_label, rec_option);

    if (!rec_model.Initialized()) {
      std::cerr << "Failed to initialize rec_model." << std::endl;
      return;
    }
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult res;
  //开始预测
  if (!ocrv3_app.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  //输出预测信息
  std::cout << res.Str() << std::endl;

  //可视化
  auto vis_img = fastdeploy::vision::Visualize::VisOcr(im_bak, res);

  cv::imwrite("vis_result.jpg", vis_img);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
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

  fastdeploy::vision::ocr::DBDetector det_model;
  fastdeploy::vision::ocr::Classifier cls_model;
  fastdeploy::vision::ocr::Recognizer rec_model;

  //准备模型
  if (!det_model_dir.empty()) {
    auto det_option = fastdeploy::RuntimeOption();
    det_option.UseGpu();
    det_option.UseTrtBackend();
    det_option.SetTrtInputShape("x", {1, 3, 50, 50}, {1, 3, 640, 640},
                                {1, 3, 960, 960});

    det_model = fastdeploy::vision::ocr::DBDetector(
        det_model_file, det_params_file, det_option);

    if (!det_model.Initialized()) {
      std::cerr << "Failed to initialize det_model." << std::endl;
      return;
    }
  }

  if (!cls_model_dir.empty()) {
    auto cls_option = fastdeploy::RuntimeOption();
    cls_option.UseGpu();
    cls_option.UseTrtBackend();
    cls_option.SetTrtInputShape("x", {1, 3, 48, 192});

    cls_model = fastdeploy::vision::ocr::Classifier(
        cls_model_file, cls_params_file, cls_option);

    if (!cls_model.Initialized()) {
      std::cerr << "Failed to initialize cls_model." << std::endl;
      return;
    }
  }

  if (!rec_model_dir.empty()) {
    auto rec_option = fastdeploy::RuntimeOption();
    rec_option.UseGpu();
    rec_option.UseTrtBackend();
    rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {1, 3, 48, 320},
                                {1, 3, 48, 2000});

    rec_model = fastdeploy::vision::ocr::Recognizer(
        rec_model_file, rec_params_file, rec_label, rec_option);

    if (!rec_model.Initialized()) {
      std::cerr << "Failed to initialize rec_model." << std::endl;
      return;
    }
  }

  auto ocrv3_app = fastdeploy::application::ocrsystem::PPOCRSystemv3(
      &det_model, &cls_model, &rec_model);

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult res;
  //开始预测
  if (!ocrv3_app.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  //输出预测信息
  std::cout << res.Str() << std::endl;

  //可视化
  auto vis_img = fastdeploy::vision::Visualize::VisOcr(im_bak, res);

  cv::imwrite("vis_result.jpg", vis_img);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
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