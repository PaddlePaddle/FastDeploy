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

 private:
  // Create op
  int width = 160;
  int height = 160;
  std::shared_ptr<fd::vision::Resize> resize_op =
      std::make_shared<fd::vision::Resize>(width, height, -1.0, -1.0, 1, false);
  std::shared_ptr<fd::vision::CenterCrop> crop =
      std::make_shared<fd::vision::CenterCrop>(50, 50);
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {0.229f, 0.224f, 0.225f};
  std::shared_ptr<fd::vision::Normalize> normalize =
      std::make_shared<fd::vision::Normalize>(mean, std);
};

// Implement our custom processor's Apply() method
bool CustomPreprocessor::Apply(fd::vision::FDMatBatch* image_batch,
                               std::vector<fd::FDTensor>* outputs) {
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
    std::cout << "Usage: ./preprocessor_demo path/to/image run_option, "
                 "e.g ././preprocessor_demo ./test.jpeg 0"
              << std::endl;
    std::cout << "Run_option 0: OpenCV; 1: CV-CUDA " << std::endl;
    return -1;
  }

  // Prepare input images
  auto im = cv::imread(argv[1]);
  std::vector<cv::Mat> images = {im, im};
  std::vector<fd::vision::FDMat> mats = fd::vision::WrapMat(images);
  std::vector<fd::FDTensor> outputs;

  // CustomPreprocessor processor;
  CustomPreprocessor processor = CustomPreprocessor();

  // Use CV-CUDA if parameter passed and detected
  if (std::atoi(argv[2]) == 1) {
    processor.UseCuda(true, 0);
  }

  // Run the processor
  bool ret = processor.Run(&mats, &outputs);

  // Print output
  for (int i = 0; i < outputs.size(); i++) {
    outputs[i].PrintInfo("out");
  }

  return 0;
}