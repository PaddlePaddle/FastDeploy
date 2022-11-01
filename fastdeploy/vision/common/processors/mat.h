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
#include "fastdeploy/core/fd_tensor.h"
#include "opencv2/core/core.hpp"
#include "fastdeploy/vision/common/processors/utils.h"
#ifdef ENABLE_OPENCV_CUDA
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#endif

namespace fastdeploy {
namespace vision {

enum class FASTDEPLOY_DECL ProcLib { DEFAULT, OPENCV, OPENCVCUDA, FLYCV };
enum Layout { HWC, CHW };

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out, const ProcLib& p);

struct FASTDEPLOY_DECL Mat {
  explicit Mat(cv::Mat& mat) {
    cpu_mat = mat;
    layout = Layout::HWC;
    height = cpu_mat.rows;
    width = cpu_mat.cols;
    channels = cpu_mat.channels();
    mat_type = ProcLib::OPENCV;
  }

  // careful if you use this interface
  // this only used if you don't want to write
  // the original data, and write to a new cv::Mat
  // then replace the old cv::Mat of this structure
  void SetMat(const cv::Mat& mat) {
    cpu_mat = mat;
    mat_type = ProcLib::OPENCV;
  }

  inline cv::Mat* GetOpenCVMat() {
    FDASSERT(mat_type == ProcLib::OPENCV, "Met non cv::Mat data structure.");
    return &cpu_mat;
  }


  inline const cv::Mat* GetOpenCVMat() const {
    FDASSERT(mat_type == ProcLib::OPENCV, "Met non cv::Mat data structure.");
    return &cpu_mat;
  }

#ifdef ENABLE_OPENCV_CUDA
  void SetMat(const cv::cuda::GpuMat& mat) {
    gpu_mat = mat;
    mat_type = ProcLib::OPENCVCUDA;
  }

  inline cv::cuda::GpuMat* GetOpenCVCudaMat() {
    FDASSERT(mat_type == ProcLib::OPENCVCUDA,
             "Met non cv::cuda::GpuMat data strucure.");
    return &gpu_mat;
  }
#endif

#ifdef ENABLE_FLYCV
  void SetMat(const fcv::Mat& mat) {
    fcv_mat = mat;
    mat_type = ProcLib::FLYCV;
  }

  inline fcv::Mat* GetFalconCVMat() {
    FDASSERT(mat_type == ProcLib::FLYCV, "Met non fcv::Mat data strucure.");
    return &fcv_mat;
  }
#endif

  void* Data();

 private:
  int channels;
  int height;
  int width;
  cv::Mat cpu_mat;

#ifdef ENABLE_OPENCV_CUDA
  cv::cuda::GpuMat gpu_mat;
#endif

#ifdef ENABLE_FLYCV
  fcv::Mat fcv_mat;
#endif

 public:
  template<typename T>
  T* GetMat() {
    return &cpu_mat;
  }

  FDDataType Type();
  int Channels() const { return channels; }
  int Width() const { return width; }
  int Height() const { return height; }
  void SetChannels(int s) { channels = s; }
  void SetWidth(int w) { width = w; }
  void SetHeight(int h) { height = h; }

  // Transfer the vision::Mat to FDTensor
  void ShareWithTensor(FDTensor* tensor);
  // Only support copy to cpu tensor now
  bool CopyToTensor(FDTensor* tensor);

  // debug functions
  // TODO(jiangjiajun) Develop a right process pipeline with c++ is not a easy
  // things
  // Will add more debug function here to help debug processed image
  // This function will print shape / mean of each channels of the Mat
  void PrintInfo(const std::string& flag);

  ProcLib mat_type = ProcLib::OPENCV;
  Layout layout = Layout::HWC;
};

Mat CreateFromTensor(const FDTensor& tensor);

}  // namespace vision
}  // namespace fastdeploy
