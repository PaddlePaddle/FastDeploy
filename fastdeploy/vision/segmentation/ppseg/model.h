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
/** \brief All segmentation model APIs are defined inside this namespace
 *
 */
namespace segmentation {

/*! @brief PaddleSeg serials model object used when to load a PaddleSeg model exported by PaddleSeg repository
 */
class FASTDEPLOY_DECL PaddleSegModel : public FastDeployModel {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g unet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g unet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g unet/deploy.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PaddleSegModel(const std::string& model_file, const std::string& params_file,
                 const std::string& config_file,
                 const RuntimeOption& custom_option = RuntimeOption(),
                 const ModelFormat& model_format = ModelFormat::PADDLE);

  /// Get model's name
  std::string ModelName() const { return "PaddleSeg"; }

  /** \brief Predict the segmentation result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread()
   * \param[in] result The output segmentation result will be writen to this structure
   * \return true if the segmentation prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, SegmentationResult* result);

  /** \brief Whether applying softmax operator in the postprocess, default value is false
   */
  bool apply_softmax = false;

  /** \brief For PP-HumanSeg model, set true if the input image is vertical image(height > width), default value is false
   */
  bool is_vertical_screen = false;

  // RKNPU2 can run normalize and hwc2chw on the NPU.
  // This function is used to close normalize and hwc2chw operations in preprocessing.
  void DisableNormalizeAndPermute();
 private:
  bool Initialize();

  bool BuildPreprocessPipelineFromConfig();

  bool Preprocess(Mat* mat, FDTensor* outputs);

  bool Postprocess(FDTensor* infer_result, SegmentationResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info);

  bool is_with_softmax = false;

  bool is_with_argmax = true;

  std::vector<std::shared_ptr<Processor>> processors_;
  std::string config_file_;
  
  // for recording the switch of normalize and hwc2chw
  bool switch_of_nor_and_per = true;  
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
