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
#include "fastdeploy/vision/common/pre_post_process_base.h"

namespace fastdeploy {
namespace vision {
/** \brief All segmentation model APIs are defined inside this namespace
 *
 */
namespace segmentation {

class FASTDEPLOY_DECL PaddleSegPreprocess : public BasePreprocess {
 public:
  PaddleSegPreprocess() {}
  PaddleSegPreprocess(const std::string& config_file,
                      std::vector<Backend>* valid_cpu_backends,
                      std::vector<Backend>* valid_gpu_backends) {
    config_file_ = config_file;
    valid_cpu_backends_ = valid_cpu_backends;
    valid_gpu_backends_ = valid_gpu_backends;
  }
  virtual bool BuildPreprocessPipelineFromConfig();
  virtual bool Run(Mat* mat, FDTensor* outputs,
                   bool is_vertical_screen);

  bool is_with_softmax_ = false;

  bool is_with_argmax_ = true;

  std::vector<Backend>* valid_cpu_backends_ = nullptr;

  std::vector<Backend>* valid_gpu_backends_ = nullptr;
};

class FASTDEPLOY_DECL PaddleSegPostprocess : public BasePostprocess {
 public:
  PaddleSegPostprocess() {}
  PaddleSegPostprocess(std::map<std::string, std::array<int, 2>> im_info,
                      const bool& is_with_argmax, const bool& is_with_softmax,
                      const bool& apply_softmax) {
    im_info_ = im_info;

    is_with_argmax_ = is_with_argmax;

    is_with_softmax_ = is_with_softmax;

    apply_softmax_ =  apply_softmax;
  }

  virtual bool Run(FDTensor* infer_result, SegmentationResult* result);

  bool is_with_softmax_ = false;

  bool is_with_argmax_ = true;

  bool apply_softmax_ = false;
};

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
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
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

 private:
  bool Initialize();

  bool Postprocess(FDTensor* infer_result, SegmentationResult* result,
                   const std::map<std::string, std::array<int, 2>>& im_info,
                   const bool& is_with_argmax, const bool& is_with_softmax);

  PaddleSegPreprocess preprocess;

  PaddleSegPostprocess postprocess;
};

}  // namespace segmentation
}  // namespace vision
}  // namespace fastdeploy
