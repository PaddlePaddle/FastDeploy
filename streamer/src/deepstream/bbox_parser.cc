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

#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

static float clamp(const float val, const float min_val, const float max_val) {
  assert(min_val <= max_val);
  return std::min(max_val, std::max(min_val, val));
}

extern "C" bool NvDsInferParseCustomPPYOLOE(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) {
  if (outputLayersInfo.empty()) {
    std::cerr << "Could not find output layer in bbox parsing" << std::endl;;
    return false;
  }

  int num_classes = outputLayersInfo[0].inferDims.d[0];
  if (num_classes != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Num classes mismatch. Configured:"
              << detectionParams.numClassesConfigured
              << ", detected by network: " << num_classes << std::endl;
    assert(-1);
  }

  int num_obj = outputLayersInfo[0].inferDims.d[1];
  float* score_data = (float*)outputLayersInfo[0].buffer;
  float* bbox_data = (float*)outputLayersInfo[1].buffer;

  for (int i = 0; i < num_obj; i++) {
    float max_score = -1.0f;
    int class_id = -1;
    for (int j = 0; j < num_classes; j++) {
      float score = score_data[num_obj * j + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }
    NvDsInferParseObjectInfo obj;
    obj.classId = (uint32_t)class_id;
    obj.detectionConfidence = max_score;
    obj.left = bbox_data[4 * i];
    obj.top = bbox_data[4 * i + 1];
    obj.width = bbox_data[4 * i + 2] - bbox_data[4 * i];
    obj.height = bbox_data[4 * i + 3] - bbox_data[4 * i + 1];
    obj.left = clamp(obj.left, 0, networkInfo.width);
    obj.top = clamp(obj.top, 0, networkInfo.height);
    obj.width = clamp(obj.width, 0, networkInfo.width);
    obj.height = clamp(obj.height, 0, networkInfo.height);
    objectList.push_back(obj);
  }
  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPPYOLOE);
