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
//构造
Classifier::Classifier(const std::string& model_file,
                       const std::string& params_file,
                       const RuntimeOption& custom_option,
                       const Frontend& model_format) {
  if (model_format == Frontend::ONNX) {
    valid_cpu_backends = {Backend::ORT};  // 指定可用的CPU后端
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  // 指定可用的GPU后端
  } else {
    // Cls模型暂不支持ORT后端推理
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::TRT, Backend::ORT};
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
  cls_thresh = 0.9;
  cls_image_shape = {3, 48, 192};
  cls_batch_num = 1;
  mean = {0.485f, 0.456f, 0.406f};
  scale = {0.5f, 0.5f, 0.5f};
  is_scale = true;

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

//预处理
//预处理
bool Classifier::Preprocess(Mat* mat, FDTensor* output) {
  // 1. cls resizes
  // 2. normalize
  // 3. batch_permute
  // for (int ino = cur_index; ino < end_img_no; ino++) {
  OcrClassifierResizeImage(mat, cls_image_shape);

  Normalize::Run(mat, mean, scale, true);

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);

  return true;
}

//后处理
bool Classifier::Postprocess(FDTensor& infer_result, int& cls_labels,
                             float& cls_scores) {
  // infer_result : n, c, h , w
  std::vector<int64_t> output_shape = infer_result.shape;
  FDASSERT(output_shape[0] == 1, "Only support batch =1 now.");

  float* out_data = static_cast<float*>(infer_result.Data());

  int label = std::distance(
      &out_data[0], std::max_element(&out_data[0], &out_data[output_shape[1]]));

  float score =
      float(*std::max_element(&out_data[0], &out_data[output_shape[1]]));

  cls_labels = label;
  cls_scores = score;

  return true;
}

//预测
bool Classifier::Predict(cv::Mat* img, int& cls_labels, float& cls_socres) {
  Mat mat(*img);
  std::vector<FDTensor> input_tensors(1);

  if (!Preprocess(&mat, &input_tensors[0])) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors[0], cls_labels, cls_socres)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;
}

}  // namesapce ocr
}  // namespace vision
}  // namespace fastdeploy