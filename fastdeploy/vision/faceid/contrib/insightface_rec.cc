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

#include "fastdeploy/vision/faceid/contrib/insightface_rec.h"

#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace faceid {

InsightFaceRecognitionModel::InsightFaceRecognitionModel(
    const std::string& model_file, const std::string& params_file,
    const RuntimeOption& custom_option, const ModelFormat& model_format) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool InsightFaceRecognitionModel::Initialize() {
  // parameters for preprocess
  size = {112, 112};
  alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  beta = {-1.f, -1.f, -1.f};  // RGB
  swap_rb = true;
  l2_normalize = false;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool InsightFaceRecognitionModel::Preprocess(Mat* mat, FDTensor* output) {
  // face recognition model's preprocess steps in insightface
  // reference: insightface/recognition/arcface_torch/inference.py
  // 1. Resize
  // 2. BGR2RGB
  // 3. Convert(opencv style) or Normalize
  // 4. HWC2CHW
  int resize_w = size[0];
  int resize_h = size[1];
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Resize::Run(mat, resize_w, resize_h);
  }
  if (swap_rb) {
    BGR2RGB::Run(mat);
  }

  Convert::Run(mat, alpha, beta);
  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool InsightFaceRecognitionModel::Postprocess(
    std::vector<FDTensor>& infer_result, FaceRecognitionResult* result) {
  FDASSERT((infer_result.size() == 1),
           "The default number of output tensor must be 1 according to "
           "insightface.");
  FDTensor& embedding_tensor = infer_result.at(0);
  FDASSERT((embedding_tensor.shape[0] == 1), "Only support batch =1 now.");
  if (embedding_tensor.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }

  result->Clear();
  result->Resize(embedding_tensor.Numel());
  // Copy the raw embedding vector directly without L2 normalize
  // post process. Let the user decide whether to normalize or not.
  // Will call utils::L2Normlize() method to perform L2
  // normalize if l2_normalize was set as 'true'.
  std::memcpy(result->embedding.data(), embedding_tensor.Data(),
              embedding_tensor.Nbytes());
  if (l2_normalize) {
    auto norm_embedding = utils::L2Normalize(result->embedding);
    std::memcpy(result->embedding.data(), norm_embedding.data(),
                embedding_tensor.Nbytes());
  }
  return true;
}

bool InsightFaceRecognitionModel::Predict(cv::Mat* im,
                                          FaceRecognitionResult* result) {
  Mat mat(*im);
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

  if (!Postprocess(output_tensors, result)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }
  return true;
}

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy