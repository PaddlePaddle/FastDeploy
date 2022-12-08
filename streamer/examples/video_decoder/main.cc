
#include "fd_streamer.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {

  auto streamer = fastdeploy::streamer::FDStreamer();
  streamer.Init("streamer_cfg.yml");
  streamer.SetupCallback();
  streamer.RunAsync();
  int count = 0;
  fastdeploy::FDTensor tensor;
  while (1) {
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    bool ret = streamer.PopTensor(tensor);
    if (!ret) continue;
    count++;
    std::cout << "main: " << count << std::endl;
    tensor.PrintInfo();
    cv::Mat mat(tensor.shape[0], tensor.shape[1], CV_8UC3, tensor.Data());
    cv::imwrite("out/" + std::to_string(count) + ".jpg", mat);
  }
  return 0;
}
