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

namespace fastdeploy {
namespace vision {
/** \brief All classification model APIs are defined inside this namespace
 *
 */
namespace structure_test {
/*! @brief SERViLayoutxlm serials model object used when to load a SERViLayoutxlm model exported by SERViLayoutxlm repository
 */
class FASTDEPLOY_DECL SERViLayoutxlmModel : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  SERViLayoutxlmModel(const std::string& model_file,
                  const std::string& params_file,
                  const std::string& config_file,
                  const RuntimeOption& custom_option = RuntimeOption(),
                  const ModelFormat& model_format = ModelFormat::PADDLE);

  /** \brief Clone a new SERViLayoutxlmModel with less memory usage when multiple instances of the same model are created
   *
   * \return new SERViLayoutxlmModel* type unique pointer
   */
  virtual std::unique_ptr<SERViLayoutxlmModel> Clone() const;

  /// Get model's name
  virtual std::string ModelName() const { return "SERViLayoutxlm"; }

 protected:
  bool Initialize();
};

}  // namespace structure_test
}  // namespace vision
}  // namespace fastdeploy
