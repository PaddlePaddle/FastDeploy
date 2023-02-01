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

  virtual bool ImplByOpenCV(Mat* mat) {
    FDERROR << Name() << " Not Implement Yet." << std::endl;
    return false;
  }

  virtual bool ImplByOpenCV(std::vector<Mat>* mats) {
    for (size_t i = 0; i < mats->size(); ++i) {
      if (ImplByOpenCV(&((*mats)[i])) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByOpenCV(MatBatch* mat_batch) {
    for (size_t i = 0; i < mat_batch->mats->size(); ++i) {
      if (ImplByOpenCV(&(*(mat_batch->mats))[i]) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByFlyCV(Mat* mat) {
    return ImplByOpenCV(mat);
  }

  virtual bool ImplByFlyCV(std::vector<Mat>* mats) {
    for (size_t i = 0; i < mats->size(); ++i) {
      if (ImplByFlyCV(&((*mats)[i])) != true) {
        return false;
      }
    }
    return true;
  }

  virtual bool ImplByCuda(Mat* mat) {
    return ImplByOpenCV(mat);
  }

  virtual bool ImplByCvCuda(Mat* mat) {
    return ImplByOpenCV(mat);
  }

  // CUDA and CV-CUDA processors cannot use for-loop to do batch processing,
  // because of the GPU tensor cache logic.
  // So if the CUDA/CV-CUDA processor's batch processing function is not
  // implemented, it will fallback to OpenCV.
  virtual bool ImplByCuda(std::vector<Mat>* mats) {
    return ImplByOpenCV(mats);
  }

  virtual bool ImplByCvCuda(std::vector<Mat>* mats) {
    return ImplByOpenCV(mats);
  }

  virtual bool ImplByCvCuda(MatBatch* mat_batch) {
    return ImplByOpenCV(mat_batch);
  }

  virtual bool operator()(Mat* mat, ProcLib lib = ProcLib::DEFAULT);

  virtual bool operator()(std::vector<Mat>* mats,
                          ProcLib lib = ProcLib::DEFAULT);

  virtual bool operator()(MatBatch* mat_batch,
                          ProcLib lib = ProcLib::DEFAULT);

 protected:
  // Update and get the cached tensor from the cached_tensors_ map.
  // The tensor is indexed by a string.
  // If the tensor doesn't exists in the map, then create a new tensor.
  // If the tensor exists and shape is getting larger, then realloc the buffer.
  // If the tensor exists and shape is not getting larger, then return the
  // cached tensor directly.
  // FDTensor* UpdateAndGetCachedTensor(
  //     const std::vector<int64_t>& new_shape, const FDDataType& data_type,
  //     const std::string& tensor_name, const Device& new_device = Device::CPU,
  //     const bool& use_pinned_memory = false);

  // // Create an input tensor on GPU and save into cached_tensors_.
  // // If the Mat is on GPU, return the mat->Tensor() directly.
  // // If the Mat is on CPU, then create a cached GPU tensor and copy the mat's
  // // CPU tensor to this new GPU tensor.
  // FDTensor* CreateCachedGpuInputTensor(Mat* mat,
  //                                      const std::string& tensor_name);

  // Create an input tensor with batch dimension
  // The mats must have same shapes.
  // FDTensor* CreateCachedGpuInputTensor(
  //   std::vector<Mat>* mats, const std::string& tensor_name);

 private:
  std::unordered_map<std::string, FDTensor> cached_tensors_;
};

}  // namespace vision
}  // namespace fastdeploy
