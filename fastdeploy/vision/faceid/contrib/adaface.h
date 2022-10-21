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
#include "fastdeploy/vision/faceid/contrib/insightface_rec.h"

namespace fastdeploy {

namespace vision {
/** \brief All object face recognition model APIs are defined inside this namespace
 *
 */
namespace faceid {
/*! @brief AdaFace model object used when to load a AdaFace model exported by AdaFacePaddleCLas.
 */
class FASTDEPLOY_DECL AdaFace : public InsightFaceRecognitionModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./adaface.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is PADDLE format
   */
  AdaFace(const std::string& model_file, const std::string& params_file = "",
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE);

  std::string ModelName() const override {
    return "Zheng-Bicheng/AdaFacePaddleCLas";
  }
  /** \brief Predict the face recognition result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output face recognition result will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  bool Predict(cv::Mat* im, FaceRecognitionResult* result) override;

 private:
  bool Initialize() override;

  bool Preprocess(Mat* mat, FDTensor* output) override;

  bool Postprocess(std::vector<FDTensor>& infer_result,
                   FaceRecognitionResult* result) override;
};

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
