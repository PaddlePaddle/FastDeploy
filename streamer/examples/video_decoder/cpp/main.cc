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

#include "fd_streamer.h"
#include "fastdeploy/utils/perf.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  auto streamer = fastdeploy::streamer::FDStreamer();
  streamer.Init("streamer_cfg.yml");
  streamer.RunAsync();
  int count = 0;
  fastdeploy::FDTensor tensor;
  fastdeploy::TimeCounter tc;
  tc.Start();
  while (1) {
    bool ret = streamer.TryPullFrame(tensor);
    if (!ret) {
      if (streamer.Destroyed()) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    count++;
    tensor.PrintInfo();
    cv::Mat mat(tensor.shape[0], tensor.shape[1], CV_8UC3, tensor.Data());
    cv::imwrite("out/" + std::to_string(count) + ".jpg", mat);
  }
  std::cout << "Total number of frames: " << count << std::endl;
  tc.End();
  tc.PrintInfo();
  return 0;
}
