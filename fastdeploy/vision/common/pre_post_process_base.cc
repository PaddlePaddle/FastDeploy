#include "fastdeploy/vision/common/pre_post_process_base.h"

namespace fastdeploy {
namespace vision {

bool BasePreprocess::Run(Mat* mat, FDTensor* output) {
  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }

  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);
  return true;
}

bool BasePreprocess::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  processors_.push_back(std::make_shared<BGR2RGB>());
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }   
};

}  // namespace vision
}  // namespace fastdeploy