#include <iostream>
#include <string>
#include "fastdeploy/vision.h"

void InferHumanPPHumansegv2Lite(const std::string& device = "cpu");

int main() {
  InferHumanPPHumansegv2Lite("npu");
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

void InferHumanPPHumansegv2Lite(const std::string& device) {
  std::string model_file =
      "./model/human_pp_humansegv2_lite_192x192_pretrained_3588/"
      "human_pp_humansegv2_lite_192x192_pretrained_3588.";
  std::string params_file;
  std::string config_file =
      "./model/human_pp_humansegv2_lite_192x192_pretrained_3588/deploy.yaml";

  fastdeploy::RuntimeOption option = GetOption(device);
  fastdeploy::ModelFormat format = GetFormat(device);
  model_file = GetModelPath(model_file, device);
  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option, format);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  auto image_file =
      "./image/test_portrait_pp_humansegv2_lite_256x144_pretrained.jpeg";
  auto im = cv::imread(image_file);

  if (device == "npu") {
    model.DisableNormalizeAndPermute();
  }

  fastdeploy::vision::SegmentationResult res;
  clock_t start = clock();
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }
  clock_t end = clock();
  auto dur = (double)(end - start);
  printf("infer_human_pp_humansegv2_lite_npu use time:%f\n",
         (dur / CLOCKS_PER_SEC));

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisSegmentation(im, res);
  cv::imwrite("human_pp_humansegv2_lite_npu_result.jpg", vis_im);
  std::cout
      << "Visualized result saved in ./human_pp_humansegv2_lite_npu_result.jpg"
      << std::endl;
}