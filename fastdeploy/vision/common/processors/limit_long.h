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

#include "fastdeploy/vision/common/processors/base.h"

namespace fastdeploy {
namespace vision {

class FASTDEPLOY_DECL LimitLong : public Processor {
 public:
  explicit LimitLong(int max_long = -1, int min_long = -1, int interp = 1) {
    max_long_ = max_long;
    min_long_ = min_long;
    interp_ = interp;
  }

  // Limit the long edge of image.
  // If the long edge is larger than max_long_, resize the long edge
  // to max_long_, while scale the short edge proportionally.
  // If the long edge is smaller than min_long_, resize the long edge
  // to min_long_, while scale the short edge proportionally.
  bool ImplByOpenCV(Mat* mat);
#ifdef ENABLE_FLYCV
  bool ImplByFalconCV(Mat* mat);
#endif
  std::string Name() { return "LimitLong"; }

  static bool Run(Mat* mat, int max_long = -1, int min_long = -1,
                  ProcLib lib = ProcLib::OPENCV);
  int GetMaxLong() const { return max_long_; }

 private:
  int max_long_;
  int min_long_;
  int interp_;
};
}  // namespace vision
}  // namespace fastdeploy
