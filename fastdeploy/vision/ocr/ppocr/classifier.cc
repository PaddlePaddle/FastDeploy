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

#include "fastdeploy/vision/ocr/ppocr/classifier.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"

namespace fastdeploy {
namespace vision {
namespace ocr {

Classifier::Classifier() {}
Classifier::Classifier(const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT,
                          Backend::OPENVINO}; 
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  initialized = Initialize();
}

// Init
bool Classifier::Initialize() {
  // pre&post process parameters
  cls_image_shape = {3, 48, 192};
  cls_thresh = 0.9;
  cls_batch_num = 1;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

void OcrClassifierResizeImage(Mat* mat,
                              const std::vector<int>& rec_image_shape) {
  int imgC = rec_image_shape[0];
  int imgH = rec_image_shape[1];
  int imgW = rec_image_shape[2];

  float ratio = float(mat->Width()) / float(mat->Height());

  int resize_w;
  if (ceilf(imgH * ratio) > imgW)
    resize_w = imgW;
  else
    resize_w = int(ceilf(imgH * ratio));

  Resize::Run(mat, resize_w, imgH);

  std::vector<float> value = {0, 0, 0};
  if (resize_w < imgW) {
    Pad::Run(mat, 0, 0, 0, imgW - resize_w, value);
  }
}

bool Classifier::Preprocess(Mat* mat, FDTensor* output , const std::vector<int>& cls_image_shape) {
  // 1. cls resizes
  // 2. normalize
  // 3. batch_permute
  OcrClassifierResizeImage(mat, cls_image_shape);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float>  scale = {0.5f, 0.5f, 0.5f};
  bool is_scale = true;
  Normalize::Run(mat, mean, scale, is_scale);

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);

  return true;
}

bool Classifier::Postprocess(std::vector<FDTensor>& infer_results,
                             std::tuple<int, float>* cls_result) {
  auto& infer_result = infer_results[0];
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");

  float* out_data = static_cast<float*>(infer_result.Data());

  int label = std::distance(
      &out_data[0], std::max_element(&out_data[0], &out_data[output_shape[1]]));

  float score =
      float(*std::max_element(&out_data[0], &out_data[output_shape[1]]));

  std::get<0>(*cls_result) = label;
  std::get<1>(*cls_result) = score;

  return true;
}

bool Classifier::Predict(cv::Mat* img, std::tuple<int, float>* cls_result) {
  Mat mat(*img);
  std::vector<FDTensor> input_tensors(1);

  if (!Preprocess(&mat, &input_tensors[0], cls_image_shape)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors, cls_result)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;
}

}  // namesapce ocr
}  // namespace vision
}  // namespace fastdeploy
