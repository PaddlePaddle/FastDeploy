#include "fastdeploy/vision/ppdet/yolov3.h"

namespace fastdeploy {
namespace vision {
namespace ppdet {

YOLOv3::YOLOv3(const std::string& model_file, const std::string& params_file,
               const std::string& config_file,
               const RuntimeOption& custom_option,
               const Frontend& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::PDINFER};
  valid_gpu_backends = {Backend::PDINFER};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool YOLOv3::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {
  int origin_w = mat->Width();
  int origin_h = mat->Height();
  mat->PrintInfo("Origin");
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
    mat->PrintInfo(processors_[i]->Name());
  }

  outputs->resize(3);
  (*outputs)[0].Allocate({1, 2}, FDDataType::FP32, "im_shape");
  (*outputs)[2].Allocate({1, 2}, FDDataType::FP32, "scale_factor");
  std::cout << "111111111" << std::endl;
  float* ptr0 = static_cast<float*>((*outputs)[0].MutableData());
  ptr0[0] = mat->Height();
  ptr0[1] = mat->Width();
  std::cout << "090909" << std::endl;
  float* ptr2 = static_cast<float*>((*outputs)[2].MutableData());
  ptr2[0] = mat->Height() * 1.0 / origin_h;
  ptr2[1] = mat->Width() * 1.0 / origin_w;
  std::cout << "88888" << std::endl;
  (*outputs)[1].name = "image";
  mat->ShareWithTensor(&((*outputs)[1]));
  // reshape to [1, c, h, w]
  (*outputs)[1].shape.insert((*outputs)[1].shape.begin(), 1);
  std::cout << "??????" << std::endl;
  return true;
}

}  // namespace ppdet
}  // namespace vision
}  // namespace fastdeploy
