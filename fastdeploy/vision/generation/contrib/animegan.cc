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

#include "fastdeploy/vision/generation/contrib/animegan.h"

namespace fastdeploy {
namespace vision {
namespace styletransfer {

AnimeGAN::AnimeGAN(const std::string& model_file, const std::string& params_file,
           const RuntimeOption& custom_option,
           const ModelFormat& model_format) {

  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER, Backend::TRT};

  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

bool AnimeGAN::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}


bool AnimeGAN::Preprocess(Mat* mat, FDTensor* output) {
  // 1. BGR2RGB
  // 2. Convert(opencv style) or Normalize
  BGR2RGB::Run(mat);
  Cast::Run(mat, "float");
  Convert::Run(mat, {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f}, {-1.f, -1.f, -1.f});
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool AnimeGAN::Postprocess(std::vector<FDTensor>& infer_results,
                           cv::Mat* result) {
  FDTensor& output_tensor = infer_results.at(0);  
  std::vector<int64_t> shape  = output_tensor.Shape(); // n, h, w, c
  Mat result_mat = Mat::Create(shape[1], shape[2], 3, FDDataType::FP32, output_tensor.Data());
  Convert::Run(&result_mat, {127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f});
  // tmp data type is float[0-1.0],convert to uint type
  auto temp = result_mat.GetOpenCVMat();
  cv::Mat res = cv::Mat::zeros(temp->size(), CV_8UC3);
  temp->convertTo(res, CV_8UC3, 1);
  res.copyTo(*result);
  return true;
}


bool AnimeGAN::Predict(cv::Mat* img, cv::Mat* result) {
  Mat mat(*img);
  std::vector<FDTensor> processed_data(1);

  if (!Preprocess(&mat, &(processed_data[0]))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  std::vector<FDTensor> infer_result(1);
  processed_data[0].name = InputInfoOfRuntime(0).name;
  // LOG(INFO) << processed_data[0].name;
  // for(auto index = processed_data[0].Shape().begin(); index < processed_data[0].Shape().end(); index++){
  //   LOG(INFO) << *index;
  // }
  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  if (!Postprocess(infer_result, result)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace styletransfer
}  // namespace vision
}  // namespace fastdeploy