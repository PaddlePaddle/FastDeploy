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

#include "fastdeploy/vision/faceid/contrib/cosface.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace faceid {

CosFace::CosFace(const std::string& model_file, const std::string& params_file,
                 const RuntimeOption& custom_option,
                 const ModelFormat& model_format)
    : InsightFaceRecognitionModel(model_file, params_file, custom_option,
                                  model_format) {
  initialized = Initialize();
}

bool CosFace::Initialize() {
 
  if (initialized) {
    // (1.1) re-init parameters for specific sub-classes
    size = {112, 112};
    alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    beta = {-1.f, -1.f, -1.f};  // RGB
    swap_rb = true;
    l2_normalize = false;
    return true;
  }
  if (!InsightFaceRecognitionModel::Initialize()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  // (2.1) re-init parameters for specific sub-classes
  size = {112, 112};
  alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  beta = {-1.f, -1.f, -1.f};  // RGB
  swap_rb = true;
  l2_normalize = false;
  return true;
}

bool CosFace::Preprocess(Mat* mat, FDTensor* output) {
  return InsightFaceRecognitionModel::Preprocess(mat, output);
}

bool CosFace::Postprocess(std::vector<FDTensor>& infer_result,
                          FaceRecognitionResult* result) {
  return InsightFaceRecognitionModel::Postprocess(infer_result, result);
}

bool CosFace::Predict(cv::Mat* im, FaceRecognitionResult* result) {
  return InsightFaceRecognitionModel::Predict(im, result);
}

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy