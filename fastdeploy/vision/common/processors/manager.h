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

namespace fastdeploy {
namespace vision {

class FASTDEPLOY_DECL ProcessorManager {
 public:
  ~ProcessorManager();

  void UseCuda(bool enable_cv_cuda = false, int gpu_id = -1);

  bool CudaUsed();

  void SetStream(Mat* mat) {
#ifdef WITH_GPU
    mat->SetStream(stream_);
#endif
  }

  void SyncStream() {
#ifdef WITH_GPU
    FDASSERT(cudaStreamSynchronize(stream_) == cudaSuccess,
             "[ERROR] Error occurs while sync cuda stream.");
#endif
  }

  int DeviceId() { return device_id_; }

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs);

  /** \brief The body of Run() function which needs to be implemented by a derived class
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  virtual bool Apply(std::vector<FDMat>* images,
                     std::vector<FDTensor>* outputs) = 0;

 protected:
  bool initialized_ = false;

 private:
#ifdef WITH_GPU
  cudaStream_t stream_ = nullptr;
#endif
  int device_id_ = -1;
};

}  // namespace vision
}  // namespace fastdeploy
