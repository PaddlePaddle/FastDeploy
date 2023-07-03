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

#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  fastdeploy::RuntimeOption option;
  option.UseGpu();
  option.UsePaddleBackend();

  std::string model_file = argv[1];
  std::string params_file = argv[2];
  std::string test_point = argv[3];

  auto model = fastdeploy::vision::perception::Centerpoint(
      model_file, params_file, "test", option, fastdeploy::ModelFormat::PADDLE);
  assert(model.Initialized());

  fastdeploy::vision::PerceptionResult res;
  if (!model.Predict(test_point, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return 1;
  }
  std::cout << "predict result:" << res.Str() << std::endl;

  return 0;
}
