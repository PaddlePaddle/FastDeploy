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

#include "fastdeploy/utils/utils.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/vision/common/processors/mat_batch.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <unordered_map>

namespace fastdeploy {
namespace vision {

/*! @brief Enable using FlyCV to process image while deploy vision models.
 * Currently, FlyCV in only available on ARM(Linux aarch64/Android), so will
 * fallback to using OpenCV in other platform
 */
FASTDEPLOY_DECL void EnableFlyCV();

/// Disable using FlyCV to process image while deploy vision models.
FASTDEPLOY_DECL void DisableFlyCV();

/*! @brief Set the cpu num threads of ProcLib.
 */
FASTDEPLOY_DECL void SetProcLibCpuNumThreads(int threads);

class FASTDEPLOY_DECL Processor {
 public:
  // default_lib has the highest priority
  // all the function in `processor` will force to use
  // default_lib if this flag is set.
  // DEFAULT means this flag is not set
  // static ProcLib default_lib;

  virtual std::string Name() = 0;

  virtual bool ImplByOpenCV(FDMat* mat) {
    FDERROR << Name() << " Not Implement Yet." << std::endl;
    return false;
  }

  virtual bool ImplByOpenCV(FDMatBatch* mat_batch) {
    for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
      if (ImplByOpenCV(&(*(mat_batch->mats))[i]) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByFlyCV(FDMat* mat) {
    return ImplByOpenCV(mat);
  }

  virtual bool ImplByFlyCV(FDMatBatch* mat_batch) {
    for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
      if (ImplByFlyCV(&(*(mat_batch->mats))[i]) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByCuda(FDMat* mat) {
    return ImplByOpenCV(mat);
  }

  virtual bool ImplByCuda(FDMatBatch* mat_batch) {
    for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
      if (ImplByCuda(&(*(mat_batch->mats))[i]) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByCvCuda(FDMat* mat) {
    return ImplByOpenCV(mat);
  }

  virtual bool ImplByCvCuda(FDMatBatch* mat_batch) {
    for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
      if (ImplByCvCuda(&(*(mat_batch->mats))[i]) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool operator()(FDMat* mat, ProcLib lib = ProcLib::DEFAULT);

  virtual bool operator()(FDMatBatch* mat_batch,
                          ProcLib lib = ProcLib::DEFAULT);
};

}  // namespace vision
}  // namespace fastdeploy
