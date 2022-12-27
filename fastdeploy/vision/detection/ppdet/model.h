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
#include "fastdeploy/vision/detection/ppdet/base.h"

namespace fastdeploy {
namespace vision {
namespace detection {

class FASTDEPLOY_DECL PicoDet : public PPDetBase {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g picodet/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g picodet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g picodet/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PicoDet(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT,
                        Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_rknpu_backends = {Backend::RKNPU2};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PicoDet"; }
};

class FASTDEPLOY_DECL PPYOLOE : public PPDetBase {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyoloe/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g picodet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g picodet/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PPYOLOE(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT,
                        Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_timvx_backends = {Backend::LITE};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PPYOLOE"; }
};

class FASTDEPLOY_DECL PPYOLO : public PPDetBase {
 public:
  /** \brief Set path of model file and configuration file, and the configuration of runtime
   *
   * \param[in] model_file Path of model file, e.g ppyolo/model.pdmodel
   * \param[in] params_file Path of parameter file, e.g ppyolo/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] config_file Path of configuration file for deployment, e.g picodet/infer_cfg.yml
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
   * \param[in] model_format Model format of the loaded model, default is Paddle format
   */
  PPYOLO(const std::string& model_file, const std::string& params_file,
          const std::string& config_file,
          const RuntimeOption& custom_option = RuntimeOption(),
          const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/PP-YOLO"; }
};

class FASTDEPLOY_DECL YOLOv3 : public PPDetBase {
 public:
  YOLOv3(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                        Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv3"; }
};

class FASTDEPLOY_DECL PaddleYOLOX : public PPDetBase {
 public:
  PaddleYOLOX(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER,
                        Backend::LITE};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOX"; }
};

class FASTDEPLOY_DECL FasterRCNN : public PPDetBase {
 public:
  FasterRCNN(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/FasterRCNN"; }
};

class FASTDEPLOY_DECL MaskRCNN : public PPDetBase {
 public:
  MaskRCNN(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/MaskRCNN"; }
};

class FASTDEPLOY_DECL SSD : public PPDetBase {
 public:
  SSD(const std::string& model_file, const std::string& params_file,
      const std::string& config_file,
      const RuntimeOption& custom_option = RuntimeOption(),
      const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::PDINFER, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER};
    valid_kunlunxin_backends = {Backend::LITE};
    valid_ascend_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/SSD"; }
};

class FASTDEPLOY_DECL PaddleYOLOv5 : public PPDetBase {
 public:
  PaddleYOLOv5(const std::string& model_file, const std::string& params_file,
               const std::string& config_file,
               const RuntimeOption& custom_option = RuntimeOption(),
               const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv5"; }
};

class FASTDEPLOY_DECL PaddleYOLOv6 : public PPDetBase {
 public:
  PaddleYOLOv6(const std::string& model_file, const std::string& params_file,
               const std::string& config_file,
               const RuntimeOption& custom_option = RuntimeOption(),
               const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv6"; }
};

class FASTDEPLOY_DECL PaddleYOLOv7 : public PPDetBase {
 public:
  PaddleYOLOv7(const std::string& model_file, const std::string& params_file,
               const std::string& config_file,
               const RuntimeOption& custom_option = RuntimeOption(),
               const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/YOLOv7"; }
};

class FASTDEPLOY_DECL RTMDet : public PPDetBase {
 public:
  RTMDet(const std::string& model_file, const std::string& params_file,
         const std::string& config_file,
         const RuntimeOption& custom_option = RuntimeOption(),
         const ModelFormat& model_format = ModelFormat::PADDLE)
      : PPDetBase(model_file, params_file, config_file, custom_option,
                model_format) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT, Backend::PDINFER};
    valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
    valid_kunlunxin_backends = {Backend::LITE};
    initialized = Initialize();
  }

  virtual std::string ModelName() const { return "PaddleDetection/RTMDet"; }
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
