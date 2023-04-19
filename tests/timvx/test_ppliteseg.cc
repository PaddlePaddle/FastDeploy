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

#include "common.h"
#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string& model_dir, const std::string& image_file,
                  const std::string& seg_result_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "deploy.yaml";
  auto subgraph_file = model_dir + sep + "subgraph.txt";
  fastdeploy::vision::EnableFlyCV();
  fastdeploy::RuntimeOption option;
  option.UseTimVX();
  option.SetLiteSubgraphPartitionPath(subgraph_file);

  auto model = fastdeploy::vision::segmentation::PaddleSegModel(
      model_file, params_file, config_file, option);

  assert(model.Initialized());

  auto im = cv::imread(image_file);

  fastdeploy::vision::SegmentationResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // std::cout << res.Str() << std::endl;
  // std::ofstream res_str(seg_result_file);
  // if(!WriteSegResult(res, seg_result_file)){
  //   std::cerr << "Fail to write to " << seg_result_file<<std::endl;
  // }
  // std::cout<<"file writen"<<std::endl;

  if (CompareSegResult(res, seg_result_file)) {
    std::cout << model_dir + " run successfully." << std::endl;
  } else {
    std::cerr << model_dir + " run failed." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: infer_demo path/to/quant_model "
                 "path/to/image "
                 "e.g ./infer_demo ./ResNet50_vd_quant ./test.jpeg "
                 "./ppliteseg_result.txt"
              << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  std::string seg_result_file = argv[3];
  InitAndInfer(model_dir, test_image, seg_result_file);
  return 0;
}
