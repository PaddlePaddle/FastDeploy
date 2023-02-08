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
  auto det_model_file = det_model_dir + sep + "ch_PP-OCRv3_det_infer.onnx";
  auto det_params_file = "";

  auto cls_model_file = cls_model_dir + sep + "ch_ppocr_mobile_v2.0_cls_infer.onnx";
  auto cls_params_file = "";

  auto rec_model_file = rec_model_dir + sep + "ch_PP-OCRv3_rec_infer.onnx";
  auto rec_params_file = "";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  auto det_model = fastdeploy::vision::ocr::DBDetector(det_model_file, det_params_file, det_option,fastdeploy::ONNX);
  auto cls_model = fastdeploy::vision::ocr::Classifier(cls_model_file, cls_params_file, cls_option,fastdeploy::ONNX);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(rec_model_file, rec_params_file, rec_label_file, rec_option,fastdeploy::ONNX);

  // Users could enable static shape infer for rec model when deploy PP-OCR on hardware 
  // which can not support dynamic shape infer well, like Huawei Ascend series. 
  rec_model.GetPreprocessor().SetStaticShapeInfer(true);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // The classification model is optional, so the PP-OCR can also be connected in series as follows
  // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // When users enable static shape infer for rec model, the batch size of cls and rec model must to be set to 1.
  ppocr_v3.SetClsBatchSize(1);
  ppocr_v3.SetRecBatchSize(1); 

  if(!ppocr_v3.Initialized()){
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  
  fastdeploy::vision::OCRResult result;
  if (!ppocr_v3.Predict(im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  fastdeploy::RuntimeOption option;
  option.UseCpu();

  std::string det_model_dir = "../ch_PP-OCRv3_det_infer";
  std::string cls_model_dir = "../ch_ppocr_mobile_v2.0_cls_infer";
  std::string rec_model_dir = "../ch_PP-OCRv3_rec_infer";
  std::string rec_label_file = "../ppocr_keys_v1.txt";
  std::string test_image = "../12.jpg";
  InitAndInfer(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file, test_image, option);
  return 0;
}
