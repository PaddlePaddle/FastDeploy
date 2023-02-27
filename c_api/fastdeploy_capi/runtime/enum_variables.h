// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#define FD_ENUM(type)                                                          \
  typedef int32_t type;                                                        \
  enum

FD_ENUM(FD_C_ModelFormat){
    FD_C_ModelFormat_AUTOREC,      ///< Auto recognize the model format by model file name
    FD_C_ModelFormat_PADDLE,       ///< Model with paddlepaddle format
    FD_C_ModelFormat_ONNX,         ///< Model with ONNX format
    FD_C_ModelFormat_RKNN,         ///< Model with RKNN format
    FD_C_ModelFormat_TORCHSCRIPT,  ///< Model with TorchScript format
    FD_C_ModelFormat_SOPHGO,       ///< Model with SOPHGO format
};

FD_ENUM(FD_C_rknpu2_CpuName){
    FD_C_ModelFormat_RK356X = 0, /* run on RK356X. */
    FD_C_ModelFormat_RK3588 = 1, /* default,run on RK3588. */
    FD_C_ModelFormat_UNDEFINED,
};

FD_ENUM(FD_C_rknpu2_CoreMask){
    FD_C_ModelFormat_RKNN_NPU_CORE_AUTO = 0,  //< default, run on NPU core randomly.
    FD_C_ModelFormat_RKNN_NPU_CORE_0 = 1,     //< run on NPU core 0.
    FD_C_ModelFormat_RKNN_NPU_CORE_1 = 2,     //< run on NPU core 1.
    FD_C_ModelFormat_RKNN_NPU_CORE_2 = 4,     //< run on NPU core 2.
    FD_C_ModelFormat_RKNN_NPU_CORE_0_1 = FD_C_ModelFormat_RKNN_NPU_CORE_0 |
                        FD_C_ModelFormat_RKNN_NPU_CORE_1,  //< run on NPU core 1 and core 2.
    FD_C_ModelFormat_RKNN_NPU_CORE_0_1_2 = FD_C_ModelFormat_RKNN_NPU_CORE_0_1 |
                          FD_C_ModelFormat_RKNN_NPU_CORE_2,  //< run on NPU core 1 and core 2.
    FD_C_ModelFormat_RKNN_NPU_CORE_UNDEFINED,
};

FD_ENUM(FD_C_LitePowerMode){
    FD_C_ModelFormat_LITE_POWER_HIGH = 0,       ///< Use Lite Backend with high power mode
    FD_C_ModelFormat_LITE_POWER_LOW = 1,        ///< Use Lite Backend with low power mode
    FD_C_ModelFormat_LITE_POWER_FULL = 2,       ///< Use Lite Backend with full power mode
    FD_C_ModelFormat_LITE_POWER_NO_BIND = 3,    ///< Use Lite Backend with no bind power mode
    FD_C_ModelFormat_LITE_POWER_RAND_HIGH = 4,  ///< Use Lite Backend with rand high mode
    FD_C_ModelFormat_LITE_POWER_RAND_LOW = 5    ///< Use Lite Backend with rand low power mode
};

FD_ENUM(FD_C_ResultType){
    FD_C_ModelFormat_UNKNOWN_RESULT,
    FD_C_ModelFormat_CLASSIFY,
    FD_C_ModelFormat_DETECTION,
    FD_C_ModelFormat_SEGMENTATION,
    FD_C_ModelFormat_OCR,
    FD_C_ModelFormat_MOT,
    FD_C_ModelFormat_FACE_DETECTION,
    FD_C_ModelFormat_FACE_ALIGNMENT,
    FD_C_ModelFormat_FACE_RECOGNITION,
    FD_C_ModelFormat_MATTING,
    FD_C_ModelFormat_MASK,
    FD_C_ModelFormat_KEYPOINT_DETECTION,
    FD_C_ModelFormat_HEADPOSE,
};
