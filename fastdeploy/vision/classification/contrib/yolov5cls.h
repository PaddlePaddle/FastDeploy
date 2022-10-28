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
/** \brief All image classification model APIs are defined inside this namespace
 *
 */
namespace classification {

/*! @brief YOLOv5Cls model object used when to load a YOLOv5Cls model exported by YOLOv5
 */
class FASTDEPLOY_DECL YOLOv5Cls : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g yolov5cls/yolov5n-cls.onnx
   * \param[in] params_file Path of parameter file, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  YOLOv5Cls(const std::string& model_file, const std::string& params_file = "",
            const RuntimeOption& custom_option = RuntimeOption(),
            const ModelFormat& model_format = ModelFormat::ONNX);

  /// Get model's name
  virtual std::string ModelName() const { return "yolov5cls"; }

  /** \brief Predict the classification result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output classification result will be writen to this structure
   * \param[in] topk Returns the topk classification result with the highest predicted probability, the default is 1
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);

  /// Preprocess image size, the default is (224, 224)
  std::vector<int> size;

 private:
  bool Initialize();
  /// Preprocess an input image, and set the preprocessed results to `outputs`
  bool Preprocess(Mat* mat, FDTensor* output,
                  const std::vector<int>& size = {224, 224});

  /// Postprocess the inferenced results, and set the final result to `result`
  bool Postprocess(const FDTensor& infer_result, ClassifyResult* result,
                   int topk = 1);
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
