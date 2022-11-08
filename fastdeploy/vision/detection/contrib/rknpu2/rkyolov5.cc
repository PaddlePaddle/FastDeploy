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

#include "rkyolov5.h"
#include <array>
namespace fastdeploy {
namespace vision {
namespace detection {

RKYOLOv5::RKYOLOv5(const std::string& model_file,
                   const std::string& params_file,
                   const fastdeploy::RuntimeOption& custom_option,
                   const fastdeploy::ModelFormat& model_format) {
  valid_cpu_backends = {Backend::ORT};
  valid_rknpu_backends = {Backend::RKNPU2};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool RKYOLOv5::Initialize() {
  // parameters for preprocess
  reused_input_tensors.resize(1);

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }

  return true;
}

bool RKYOLOv5::Preprocess(fastdeploy::vision::Mat* mat,
                          std::vector<FDTensor>* outputs) {

  std::vector<float> input_shape = {static_cast<float>(mat->Height()),
                                    static_cast<float>(mat->Width())};
  std::vector<float> output_shape = {640.0, 640.0};
  std::vector<float> padding_value = {114.0, 114.0, 114.0};

  // process after image load
  float ratio = (output_shape[0]) / std::max(static_cast<float>(mat->Height()),
                                             static_cast<float>(mat->Width()));
  if (ratio != 1.0) {
    int interp = cv::INTER_AREA;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Resize::Run(mat, resize_w, resize_h, -1, -1, interp);
  }

  // yolov5's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  PadToSize::Run(mat, output_shape[0], output_shape[1], padding_value);
  BGR2RGB::Run(mat);
//  HWC2CHW::Run(mat);
//  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
//  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
//  Convert::Run(mat, alpha, beta);
  Cast::Run(mat, "float");

  outputs->resize(1);
  (*outputs)[0].name = InputInfoOfRuntime(0).name;
  mat->ShareWithTensor(&((*outputs)[0]));
  // reshape to [1, c, h, w]
  (*outputs)[0].shape.insert((*outputs)[0].shape.begin(), 1);
  return true;
}

int RKYOLOv5::ArgMax(std::vector<float>& vSingleProbs) {
  int result;
  auto iter = std::max_element(vSingleProbs.begin(), vSingleProbs.end());
  result = static_cast<int>(iter - vSingleProbs.begin());
  return result;
}

bool RKYOLOv5::Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result) {
  int nc = 80;
  float* output = static_cast<float*>(infer_result[0].Data());
  float confThresh = 0.2;
  int num_box = 0;
  for (int i = 0; i < 25200; i++) {
    float conf = output[i * (nc + 5) + 4];
    if (conf > confThresh) {
      num_box++;
    }
  }
  result->Reserve(num_box);

  for (int i = 0; i < 25200; i++) {
    float conf = output[i * (nc + 5) + 4];
    if (conf > confThresh) {
      float cx = output[i * (nc + 5)];
      float cy = output[i * (nc + 5) + 1];
      float w = output[i * (nc + 5) + 2];
      float h = output[i * (nc + 5) + 3];
      std::vector<float> vSingleProbs(nc);
      for (int j = 0; j < vSingleProbs.size(); j++) {
        vSingleProbs[j] = output[i * 85 + 5 + j];
      }
      result->label_ids.push_back(ArgMax(vSingleProbs));
      result->scores.push_back(conf);
      result->boxes.emplace_back(std::array<float, 4>{
          cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f});
    }
  }
  utils::NMS(result);
  return true;
}

bool RKYOLOv5::Predict(cv::Mat* im, DetectionResult* result) {
  Mat mat(*im);

  std::vector<FDTensor> processed_data;
  if (!Preprocess(&mat, &processed_data)) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  float* tmp = static_cast<float*>(processed_data[1].Data());
  std::vector<FDTensor> infer_result;
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
} // namespace detection
} // namespace vision
} // namespace fastdeploy