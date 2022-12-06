#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>


static const int NUM_CLASSES_YOLO = 80;

static float clamp(const float val, const float minVal, const float maxVal) {
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
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
  // std::cout << "num of layers: " << outputLayersInfo.size() << std::endl;
  // std::cout << "num of layers: " << outputLayersInfo[0].layerName << std::endl;
  // std::cout << "num of layers: " << outputLayersInfo[1].layerName << std::endl;

  if (NUM_CLASSES_YOLO != detectionParams.numClassesConfigured) {
    std::cerr << "WARNING: Num classes mismatch. Configured:"
              << detectionParams.numClassesConfigured
              << ", detected by network: " << NUM_CLASSES_YOLO << std::endl;
  }

  float* score_data = (float*)outputLayersInfo[0].buffer;
  float* bbox_data = (float*)outputLayersInfo[1].buffer;
  // const int dimensions = layer.inferDims.d[1];
  // int rows = layer.inferDims.numElements / layer.inferDims.d[1];

  // std::cout << outputLayersInfo[0].inferDims.numDims << std::endl;
  // std::cout << outputLayersInfo[0].inferDims.numElements << " " << outputLayersInfo[0].inferDims.d[0] << " " << outputLayersInfo[0].inferDims.d[1] << std::endl;

  for (int i = 0; i < 8400; i++) {
    float max_score = -1.0f;
    int class_id = -1;
    for (int j = 0; j < 80; j++) {
      float score = score_data[8400 * j + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }
    // if (max_score < conf_thresh) continue;
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
