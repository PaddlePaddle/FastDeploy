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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy{

namespace vision{

namespace facedet{

class FASTDEPLOY_DECL Yolov7FacePreprocessor{

  public:
  explicit Yolov7FacePreprocessor();

  bool Run(std::vector<FDMat>* images, std::vector<FDTensor>* outputs,
           std::vector<std::map<std::string, std::array<float, 2>>>* ims_info);

  protected:
  bool Preprocess (FDMat * mat, FDTensor* output,
                   std::map<std::string, std::array<float, 2>>* im_info);

  void LetterBox(FDMat* mat);

  // target size, tuple of (width, height), default size = {640, 640}
  std::vector<int> size_;

  // padding value, size should be the same as channels
  std::vector<float> padding_color_value_;

  // only pad to the minimum rectange which height and width is times of stride
  bool is_mini_pad_;

  // while is_mini_pad = false and is_no_pad = true,
  // will resize the image to the set size
  bool is_no_pad_;

  // if is_scale_up is false, the input image only can be zoom out,
  // the maximum resize scale cannot exceed 1.0
  bool is_scale_up_;

  // padding stride, for is_mini_pad
  int stride_;

  float max_wh_;

};

}

}

}