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

#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace utils {

void FDTensor2FP32CVMat(cv::Mat& mat, FDTensor& infer_result,
                        bool contain_score_map) {
  // output with argmax channel is 1
  int channel = 1;
  int height = infer_result.shape[1];
  int width = infer_result.shape[2];

  if (contain_score_map) {
    // output without argmax and convent to NHWC
    channel = infer_result.shape[3];
  }
  // create FP32 cvmat
  if (infer_result.dtype == FDDataType::INT64) {
    FDWARNING << "The PaddleSeg model is exported with argmax. Inference "
                 "result type is " +
                     Str(infer_result.dtype) +
                     ". If you want the edge of segmentation image more "
                     "smoother. Please export model with --without_argmax "
                     "--with_softmax."
              << std::endl;
    int64_t chw = channel * height * width;
    int64_t* infer_result_buffer = static_cast<int64_t*>(infer_result.Data());
    std::vector<float_t> float_result_buffer(chw);
    mat = cv::Mat(height, width, CV_32FC(channel));
    int index = 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        mat.at<float_t>(i, j) =
            static_cast<float_t>(infer_result_buffer[index++]);
      }
    }
  } else if (infer_result.dtype == FDDataType::FP32) {
    mat = cv::Mat(height, width, CV_32FC(channel), infer_result.Data());
  }
}

}  // namespace utils
}  // namespace vision
}  // namespace fastdeploy
