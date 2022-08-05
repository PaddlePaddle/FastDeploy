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

#include "fastdeploy/vision/deepinsight/partial_fc.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {

namespace vision {

namespace deepinsight {

PartialFC::PartialFC(const std::string& model_file,
                     const std::string& params_file,
                     const RuntimeOption& custom_option,
                     const Frontend& model_format)
    : InsightFaceRecognitionModel(model_file, params_file, custom_option,
                                  model_format) {
  initialized = Initialize();
}

bool PartialFC::Initialize() {
  // 如果初始化有变化 修改该子类函数
  // 这里需要判断backend是否已经initialized，如果是，则不应该再调用
  // InsightFaceRecognitionModel::Initialize()
  // 因为该函数会对backend进行初始化, backend已经在父类的构造函数初始化
  // 这里只修改一些模型相关的属性

  // (1) 如果父类初始化了backend
  if (initialized) {
    // (1.1) re-init parameters for specific sub-classes
    size = {112, 112};
    alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    beta = {-1.f, -1.f, -1.f};  // RGB
    swap_rb = true;
    l2_normalize = false;
    return true;
  }
  // (2) 如果父类没有初始化backend
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

bool PartialFC::Preprocess(Mat* mat, FDTensor* output) {
  // 如果预处理有变化 修改该子类函数
  return InsightFaceRecognitionModel::Preprocess(mat, output);
}

bool PartialFC::Postprocess(std::vector<FDTensor>& infer_result,
                            FaceRecognitionResult* result) {
  // 如果后处理有变化 修改该子类函数
  return InsightFaceRecognitionModel::Postprocess(infer_result, result);
}

bool PartialFC::Predict(cv::Mat* im, FaceRecognitionResult* result) {
  // 如果前后处理有变化 则override子类的Preprocess和Postprocess
  // 如果前后处理有变化 此处应该调用子类自己的Preprocess和Postprocess
  return InsightFaceRecognitionModel::Predict(im, result);
}

}  // namespace deepinsight
}  // namespace vision
}  // namespace fastdeploy