#pragma once
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/vision/common/processors/mat.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "yaml-cpp/yaml.h"
#include <map>

namespace fastdeploy {
namespace vision {

class FASTDEPLOY_DECL BasePreprocess{
 public:
  /// Build the preprocess pipeline from the loaded model
  virtual bool BuildPreprocessPipelineFromConfig() = 0;

  virtual bool Run(Mat* mat, FDTensor* output);

  std::map<std::string, std::array<int, 2>> im_info_;

  std::vector<std::shared_ptr<Processor>> processors_;

  std::string config_file_;
};

class FASTDEPLOY_DECL BasePostprocess{
 public:
  std::map<std::string, std::array<int, 2>> im_info_;
};
}  // namespace vision
}  // namespace fastdeploy
