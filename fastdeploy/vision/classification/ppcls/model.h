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

#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {
namespace classification {

class FASTDEPLOY_DECL PaddleClasModel : public FastDeployModel {
 public:
  PaddleClasModel(const std::string& model_file, const std::string& params_file,
        const std::string& config_file,
        const RuntimeOption& custom_option = RuntimeOption(),
        const Frontend& model_format = Frontend::PADDLE);

  virtual std::string ModelName() const { return "PaddleClas/Model"; }

  // TODO(jiangjiajun) Batch is on the way
  virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);

 protected:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool Postprocess(const FDTensor& infer_result, ClassifyResult* result,
                   int topk = 1);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
};

typedef PaddleClasModel PPLCNet;
typedef PaddleClasModel PPLCNetv2;
typedef PaddleClasModel EfficientNet;
typedef PaddleClasModel GhostNet;
typedef PaddleClasModel MobileNetv1;
typedef PaddleClasModel MobileNetv2;
typedef PaddleClasModel MobileNetv3;
typedef PaddleClasModel ShuffleNetv2;
typedef PaddleClasModel SqueezeNet;
typedef PaddleClasModel Inceptionv3;
typedef PaddleClasModel PPHGNet;
typedef PaddleClasModel ResNet50vd;
typedef PaddleClasModel SwinTransformer;
}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
