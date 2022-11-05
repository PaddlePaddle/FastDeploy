#include <iostream>
#include <string>
#include "fastdeploy/vision.h"

void InferScrfd(const std::string& device = "cpu");

int main() {
  InferScrfd("npu");
  return 0;
}

fastdeploy::RuntimeOption GetOption(const std::string& device) {
  auto option = fastdeploy::RuntimeOption();
  if (device == "npu") {
    option.UseRKNPU2();
  } else {
    option.UseCpu();
  }
  return option;
}

fastdeploy::ModelFormat GetFormat(const std::string& device) {
  auto format = fastdeploy::ModelFormat::ONNX;
  if (device == "npu") {
    format = fastdeploy::ModelFormat::RKNN;
  } else {
    format = fastdeploy::ModelFormat::ONNX;
  }
  return format;
}

std::string GetModelPath(std::string& model_path, const std::string& device) {
  if (device == "npu") {
    model_path += "rknn";
  } else {
    model_path += "onnx";
  }
  return model_path;
}

void InferScrfd(const std::string& device) {
  std::string model_file =
      "./model/scrfd_500m_bnkps_shape640x640_rk3588.";
  std::string params_file;

  fastdeploy::RuntimeOption option = GetOption(device);
  fastdeploy::ModelFormat format = GetFormat(device);
  model_file = GetModelPath(model_file, device);
  auto model = fastdeploy::vision::facedet::SCRFD(
      model_file, params_file, option, format);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  auto image_file =
      "./images/test_lite_face_detector_3.jpg";
  auto im = cv::imread(image_file);

  if (device == "npu") {
    model.DisableNormalizeAndPermute();
  }

  fastdeploy::vision::FaceDetectionResult res;
  clock_t start = clock();
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  clock_t end = clock();
  auto dur = static_cast<double>(end - start);
  printf("InferScrfd use time:%f\n",
         (dur / CLOCKS_PER_SEC));

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::Visualize::VisFaceDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}