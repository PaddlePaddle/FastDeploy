#include "fastdeploy/vision.h"
#include "fastdeploy/vision/common/processors/manager.h"
#include "fastdeploy/vision/common/processors/transform.h"

namespace fd = fastdeploy;

// Define our custom processor
class CustomPreprocessor : public fd::vision::ProcessorManager {
 public:
  explicit CustomPreprocessor(){};
  ~CustomPreprocessor(){};

  virtual bool Apply(fd::vision::FDMatBatch* image_batch,
                     std::vector<fd::FDTensor>* outputs);

 protected:
  int width = 160;
  int height = 160;
};

// Implement our custom processor's Apply() method
bool CustomPreprocessor::Apply(fd::vision::FDMatBatch* image_batch,
                               std::vector<fd::FDTensor>* outputs) {
  // Create op
  auto resize_op =
      std::make_shared<fd::vision::Resize>(width, height, -1.0, -1.0, 1, false);
  auto crop = std::make_shared<fd::vision::CenterCrop>(50, 50);
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {0.229f, 0.224f, 0.225f};
  auto normalize = std::make_shared<fd::vision::Normalize>(mean, std);

  // Use op to transform the images
  bool resize_ret = (*resize_op)(&(image_batch->mats->at(0)));
  bool crop_ret = (*crop)(image_batch);
  bool normalize_ret = (*normalize)(image_batch);

  outputs->resize(1);
  fd::FDTensor* tensor = image_batch->Tensor();
  (*outputs)[0].SetExternalData(tensor->Shape(), tensor->Dtype(),
                                tensor->Data(), tensor->device,
                                tensor->device_id);
  return true;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: image_decoder path/to/image run_option, "
                 "e.g ./image_decoder ./test.jpeg 0"
              << std::endl;
    std::cout << "Run_option 0: OpenCV; 1: CVCUDA " << std::endl;
    return -1;
  }

  // Prepare input images
  auto im = cv::imread(argv[1]);
  std::vector<cv::Mat> images = {im, im};
  std::vector<fd::vision::FDMat> mats = fd::vision::WrapMat(images);
  std::vector<fd::FDTensor> outputs;

  // CustomPreprocessor processor;
  CustomPreprocessor processor = CustomPreprocessor();

  // Use CVCUDA if parameter passed and detected
  if (std::atoi(argv[2]) == 1) {
    processor.UseCuda(true, 0);
  }

  // Run the processor
  bool ret = processor.Run(&mats, &outputs);

  // Print output
  for (int i = 0; i < outputs.size(); i++) {
    outputs.at(i).PrintInfo("out");
  }

  return 0;
}