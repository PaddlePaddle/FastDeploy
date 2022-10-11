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

#include "fastdeploy/vision/classification/contrib/resnet.h"
#include "fastdeploy/vision/utils/utils.h"
#include "fastdeploy/utils/perf.h"

namespace fastdeploy {
namespace vision {
namespace classification {

ResNet::ResNet(const std::string& model_file,
               const std::string& params_file,
               const RuntimeOption& custom_option,
               const ModelFormat& model_format) {

  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT, Backend::OPENVINO}; 
    valid_gpu_backends = {Backend::ORT, Backend::TRT};  
  } else {
    valid_cpu_backends = {Backend::PDINFER};
    valid_gpu_backends = {Backend::PDINFER};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool ResNet::Initialize() {

  size = {224, 224};
  mean_vals = {0.485f, 0.456f, 0.406f};
  std_vals = {0.229f, 0.224f, 0.225f};

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}


bool ResNet::Preprocess(Mat* mat, FDTensor* output) {

// Resize
// BGR2RGB
// Normalize         
  if (mat->Height()!=size[0] || mat->Width()!=size[1]){
    int interp = cv::INTER_LINEAR;
    Resize::Run(mat, size[1], size[0], -1, -1, interp);
  }

  BGR2RGB::Run(mat);
  Normalize::Run(mat, mean_vals, std_vals);

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool ResNet::Postprocess(FDTensor& infer_result,
                                  ClassifyResult* result, int topk) {
  int num_classes = infer_result.shape[1];
  // FDTensor *softmax_res;
  Softmax(infer_result, &infer_result);
  // const float* infer_result_buffer =
  //     reinterpret_cast<const float*>(infer_result.Data());
  const float* infer_result_buffer = reinterpret_cast<float*>(infer_result.Data());
  topk = std::min(num_classes, topk);
  result->label_ids =
      utils::TopKIndices(infer_result_buffer, num_classes, topk);
  result->scores.resize(topk);
  for (int i = 0; i < topk; ++i) {
    result->scores[i] = *(infer_result_buffer + result->label_ids[i]);
  }
  return true;
}

bool ResNet::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  Mat mat(*im);
  std::vector<FDTensor> processed_data(1);
  if (!Preprocess(&mat, &(processed_data[0]))) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }
  processed_data[0].name = InputInfoOfRuntime(0).name;

  std::vector<FDTensor> output_tensors;
  if (!Infer(processed_data, &output_tensors)) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors[0], result, topk)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  return true;
}


}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
