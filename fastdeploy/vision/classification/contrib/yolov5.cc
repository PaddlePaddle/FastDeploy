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

#include "fastdeploy/vision/detection/contrib/yolov5.h"

#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace classification {

YOLOv5Cls::YOLOv5Cls(const std::string& model_file,
                     const std::string& params_file,
                     const RuntimeOption& custom_option,
                     const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool YOLOv5Cls::Initialize() {
  // preprocess parameters
  size = {224, 224};
  mean = {0.485f, 0.456f, 0.406f};
  std = {0.229f, 0.224f, 0.225f};
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool YOLOv5Cls::Preprocess(Mat* mat, FDTensor* output,
                           const std::vector<int>& size) {
  // CenterCrop
  int crop_size = min(mat->Height(), mat->Width());
  CenterCrop::Run(mat, crop_size, crop_size);
  Resize::Run(mat, size[0], size[1], -1, -1, cv::INTER_LINEAR);
  // Normalize
  HWC2CHW::Run(mat);
  BGR2RGB::Run(mat);
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);
  Normalize::Run(mat, mean, std, false);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  return true;
}

bool YOLOv5Cls::Postprocess(const FDTensor& infer_result,
                            ClassifyResult* result, int topk) {
  int num_classes = infer_result.shape[1];
  const float* infer_result_buffer =
      reinterpret_cast<const float*>(infer_result.Data());
  topk = std::min(num_classes, topk);
  result->label_ids =
      utils::TopKIndices(infer_result_buffer, num_classes, topk);
  result->scores.resize(topk);
  for (int i = 0; i < topk; ++i) {
    result->scores[i] = *(infer_result_buffer + result->label_ids[i]);
  }
  return true;
}

bool YOLOv5Cls::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  Mat mat(*im);
  std::vector<FDTensor> input_tensors(1);
  if (!Preprocess(&mat, &input_tensors[0], size)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(infer_result[0], result, topk)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
