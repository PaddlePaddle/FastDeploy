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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
/** \brief All object detection model APIs are defined inside this namespace
 *
 */
namespace detection {

/*! @brief PPYOLOE model object used when to load a PPYOLOE model exported by PaddleDetection
 */
class FASTDEPLOY_DECL PPYOLOE : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyoloe/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PPYOLOE(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE);

  /// Get model's name
  virtual std::string ModelName() const { return "PaddleDetection/PPYOLOE"; }

  /** \brief Predict the detection result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output detection result will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, DetectionResult* result);

 protected:
  PPYOLOE() {}
  virtual bool Initialize();
  /// Build the preprocess pipeline from the loaded model
  virtual bool BuildPreprocessPipelineFromConfig();
  /// Preprocess an input image, and set the preprocessed results to `outputs`
  virtual bool Preprocess(Mat* mat, std::vector<FDTensor>* outputs);

  /// Postprocess the inferenced results, and set the final result to `result`
  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           DetectionResult* result);

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  // configuration for nms
  int64_t background_label = -1;
  int64_t keep_top_k = 300;
  float nms_eta = 1.0;
  float nms_threshold = 0.7;
  float score_threshold = 0.01;
  int64_t nms_top_k = 10000;
  bool normalized = true;
  bool has_nms_ = true;

  // This function will used to check if this model contains multiclass_nms
  // and get parameters from the operator
  void GetNmsInfo();
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
