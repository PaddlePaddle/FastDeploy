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
/** \brief All classification model APIs are defined inside this namespace
 *
 */
namespace classification {
/*! @brief PaddleClas serials model object used when to load a PaddleClas model exported by PaddleClas repository
 */
class FASTDEPLOY_DECL PaddleClasModel : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PaddleClasModel(const std::string& model_file, const std::string& params_file,
                  const std::string& config_file,
                  const RuntimeOption& custom_option = RuntimeOption(),
                  const ModelFormat& model_format = ModelFormat::PADDLE);

  /// Get model's name
  virtual std::string ModelName() const { return "PaddleClas/Model"; }

  /** \brief Predict the classification result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output classification result will be writen to this structure
   * \param[in] topk (int)The topk result by the classify confidence score, default 1
   * \return true if the prediction successed, otherwise false
   */
  // TODO(jiangjiajun) Batch is on the way
  virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);

 protected:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool CudaPreprocess(Mat* mat, FDTensor* outputs);

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
