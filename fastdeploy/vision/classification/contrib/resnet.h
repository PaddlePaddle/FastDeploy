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

// The namespace shoulde be
// fastdeploy::vision::classification (fastdeploy::vision::${task})
namespace fastdeploy {
namespace vision {
/** \brief All object classification model APIs are defined inside this namespace
 *
 */
namespace classification {
/*! @brief ResNet series model
 */
class FASTDEPLOY_DECL ResNet : public FastDeployModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./resnet50.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  ResNet(const std::string& model_file,
               const std::string& params_file = "",
               const RuntimeOption& custom_option = RuntimeOption(),
               const ModelFormat& model_format = ModelFormat::ONNX);

  virtual std::string ModelName() const { return "ResNet"; }
   /** \brief Predict for the input "im", the result will be saved in "result".
   *
   * \param[in] im Input image for inference.
   * \param[in] result Saving the inference result.
   * \param[in] topk The length of return values, e.g., if topk==2, the result will include the 2 most possible class label for input image.
   */
  virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);

  /// Tuple of (width, height)
  std::vector<int> size;
  /// Mean parameters for normalize
  std::vector<float> mean_vals;
  /// Std parameters for normalize
  std::vector<float> std_vals;


 private:
  /*! @brief Initialize for ResNet model, assign values to the global variables and call InitRuntime()
  */
  bool Initialize();
  /// PreProcessing for the input "mat", the result will be saved in "outputs".
  bool Preprocess(Mat* mat, FDTensor* outputs);
  /*! @brief PostProcessing for the input "infer_result", the result will be saved in "result".
  */
  bool Postprocess(FDTensor& infer_result, ClassifyResult* result,
                   int topk = 1);
};
}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
