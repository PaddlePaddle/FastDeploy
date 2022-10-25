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

#include "fastdeploy/vision/common/processors/cast.h"
#include "fastdeploy/vision/common/processors/center_crop.h"
#include "fastdeploy/vision/common/processors/color_space_convert.h"
#include "fastdeploy/vision/common/processors/convert.h"
#include "fastdeploy/vision/common/processors/crop.h"
#include "fastdeploy/vision/common/processors/hwc2chw.h"
#include "fastdeploy/vision/common/processors/limit_by_stride.h"
#include "fastdeploy/vision/common/processors/limit_long.h"
#include "fastdeploy/vision/common/processors/limit_short.h"
#include "fastdeploy/vision/common/processors/normalize.h"
#include "fastdeploy/vision/common/processors/pad.h"
#include "fastdeploy/vision/common/processors/pad_to_size.h"
#include "fastdeploy/vision/common/processors/resize.h"
#include "fastdeploy/vision/common/processors/resize_by_long.h"
#include "fastdeploy/vision/common/processors/resize_by_short.h"
#include "fastdeploy/vision/common/processors/stride_pad.h"
#include "fastdeploy/vision/common/processors/normalize_and_permute.h"
#include "fastdeploy/vision/common/processors/warp_affine.h"

namespace fastdeploy {
namespace vision {

inline void FuseTransforms(
    std::vector<std::shared_ptr<Processor>>* processors) {
  // Fuse Normalize and HWC2CHW
  int hwc2chw_index = 0;
  for (size_t i = 0; i < processors->size(); ++i) {
    if ((*processors)[i]->Name() == "HWC2CHW") {
      if (i == 0) {
        return;
      }
      if ((*processors)[i]->Name() != "Normalize") {
        return;
      }
      hwc2chw_index = i;
    }
  }

  // Get alpha and beta of Normalize
  std::vector<float> alpha = dynamic_cast<Normalize*>(
                    (*processors)[hwc2chw_index - 1].get())->GetAlpha();
  std::vector<float> beta = dynamic_cast<Normalize*>(
                    (*processors)[hwc2chw_index - 1].get())->GetBeta();

  // Delete Normalize and HWC2CHW
  processors->erase(processors->begin() + hwc2chw_index);
  processors->erase(processors->begin() + hwc2chw_index - 1);

  // Add NormalizeAndPermute
  std::vector<float> mean({0.0, 0.0, 0.0});
  std::vector<float> std({1.0, 1.0, 1.0});
  processors->push_back(std::make_shared<NormalizeAndPermute>(mean, std));

  // Set alpha and beta
  auto processor = dynamic_cast<NormalizeAndPermute*>(
                (*processors)[hwc2chw_index - 1].get());

  processor->SetAlpha(alpha);
  processor->SetBeta(beta);
  FDINFO << "Normalize and HWC2CHW are fused to NormalizeAndPermute."
         << std::endl;
}

}  // namespace vision
}  // namespace fastdeploy
