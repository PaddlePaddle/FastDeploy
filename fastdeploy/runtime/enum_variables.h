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

/*! \file runtime.h
    \brief A brief file description.

    More details
 */

#pragma once
#include "fastdeploy/utils/utils.h"
#include <ostream>

namespace fastdeploy {

/*! Inference backend supported in FastDeploy */
enum Backend {
  UNKNOWN,  ///< Unknown inference backend
  ORT,  ///< ONNX Runtime, support Paddle/ONNX format model, CPU / Nvidia GPU
  TRT,  ///< TensorRT, support Paddle/ONNX format model, Nvidia GPU only
  PDINFER,  ///< Paddle Inference, support Paddle format model, CPU / Nvidia GPU
  POROS,    ///< Poros, support TorchScript format model, CPU / Nvidia GPU
  OPENVINO,   ///< Intel OpenVINO, support Paddle/ONNX format, CPU only
  LITE,       ///< Paddle Lite, support Paddle format model, ARM CPU only
  RKNPU2,     ///< RKNPU2, support RKNN format model, Rockchip NPU only
  SOPHGOTPU,  ///< SOPHGOTPU, support SOPHGO format model, Sophgo TPU only
};

enum FASTDEPLOY_DECL Device {CPU, GPU, RKNPU, IPU, TIMVX, KUNLUNXIN, ASCEND,
                              SOPHGOTPUD};


/*! Deep learning model format */
enum ModelFormat {
  AUTOREC,      ///< Auto recognize the model format by model file name
  PADDLE,       ///< Model with paddlepaddle format
  ONNX,         ///< Model with ONNX format
  RKNN,         ///< Model with RKNN format
  TORCHSCRIPT,  ///< Model with TorchScript format
  SOPHGO,       ///< Model with SOPHGO format
};

FASTDEPLOY_DECL std::string Str(const Device& d);

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out,
                                         const Backend& backend);


FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out, const Device& d);

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out,
                                         const ModelFormat& format);


}  // namespace fastdeploy
