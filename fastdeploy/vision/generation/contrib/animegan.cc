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
#include "fastdeploy/function/functions.h"

namespace fastdeploy {
namespace vision {
namespace generation {

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


bool AnimeGAN::Preprocess(std::vector<Mat>& images, std::vector<FDTensor>* outputs) {
  // 1. BGR2RGB
  // 2. Convert(opencv style) or Normalize
  for (size_t i = 0; i < images.size(); ++i) {
      auto ret = BGR2RGB::Run(&images[i]);
      if (!ret) {
        FDERROR << "Failed to processs image:" << i << " in "
                << "BGR2RGB" << "." << std::endl;
        return false;
      }
      ret = Cast::Run(&images[i], "float");
      if (!ret) {
        FDERROR << "Failed to processs image:" << i << " in "
                << "Cast" << "." << std::endl;
        return false;
      }
      ret = Convert::Run(&images[i], {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f}, {-1.f, -1.f, -1.f});
      if (!ret) {
        FDERROR << "Failed to processs image:" << i << " in "
                << "Cast" << "." << std::endl;
        return false;
      }
    }
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<FDTensor> tensors(images.size()); 
  for (size_t i = 0; i < images.size(); ++i) {
    images[i].ShareWithTensor(&(tensors[i]));
    tensors[i].ExpandDim(0);
  }
  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

bool AnimeGAN::Postprocess(std::vector<FDTensor>& infer_results,
                           std::vector<cv::Mat>* results) {
  // 1. Reverse normalization
  // 2. RGB2BGR
  FDTensor& output_tensor = infer_results.at(0);  
  std::vector<int64_t> shape  = output_tensor.Shape(); // n, h, w, c
  int size = shape[1] * shape[2] * shape[3];
  results->resize(shape[0]);
  float* infer_result_data = reinterpret_cast<float*>(output_tensor.Data());
  for(size_t i = 0; i < results->size(); ++i){
  float* data = new float[shape[1]*shape[2]*3];
  std::memcpy(reinterpret_cast<char*>(data), reinterpret_cast<char*>(infer_result_data+i*size), sizeof(float)*shape[1]*shape[2]*3);
  Mat result_mat = Mat::Create(shape[1], shape[2], 3, FDDataType::FP32, data);
  Convert::Run(&result_mat, {127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f});
  // tmp data type is float[0-1.0],convert to uint type
  auto temp = result_mat.GetOpenCVMat();
  cv::Mat res = cv::Mat::zeros(temp->size(), CV_8UC3);
  temp->convertTo(res, CV_8UC3, 1);
  Mat fd_image = WrapMat(res);
  BGR2RGB::Run(&fd_image);
  res = *(fd_image.GetOpenCVMat());
  res.copyTo(results->at(i));
  }
  return true;
}


bool AnimeGAN::Predict(cv::Mat& img, cv::Mat* result) {
  std::vector<cv::Mat> results;
  if (!BatchPredict({img}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool AnimeGAN::BatchPredict(const std::vector<cv::Mat>& images, std::vector<cv::Mat>* results) {
  std::vector<FDMat> fd_images = WrapMat(images);
  std::vector<FDTensor> processed_data(1);
  if (!Preprocess(fd_images, &(processed_data))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  std::vector<FDTensor> infer_result(1);
  processed_data[0].name = InputInfoOfRuntime(0).name;

  if (!Infer(processed_data, &infer_result)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }
  if (!Postprocess(infer_result, results)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace generation
}  // namespace vision
}  // namespace fastdeploy