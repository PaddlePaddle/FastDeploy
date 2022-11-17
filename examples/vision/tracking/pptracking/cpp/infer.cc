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

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const std::string& model_dir, const std::string& video_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";
  auto model = fastdeploy::vision::tracking::PPTracking(
      model_file, params_file, config_file);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  fastdeploy::vision::MOTResult result;
  fastdeploy::vision::tracking::TrailRecorder recorder;
  // during each prediction, data is inserted into the recorder. As the number of predictions increases,
  // the memory will continue to grow. You can cancel the insertion through 'UnbindRecorder'.
  // int count = 0; // unbind condition
  model.BindRecorder(&recorder);
  cv::Mat frame;
  cv::VideoCapture capture(video_file);
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    if (!model.Predict(&frame, &result)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
    // such as adding this code can cancel trail datat bind
    // if(count++ == 10) model.UnbindRecorder();
    // std::cout << result.Str() << std::endl;
    cv::Mat out_img = fastdeploy::vision::VisMOT(frame, result, 0.0, &recorder);
    cv::imshow("mot",out_img);
    cv::waitKey(30);
  }
  model.UnbindRecorder();
  capture.release();
  cv::destroyAllWindows();
}

void GpuInfer(const std::string& model_dir, const std::string& video_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto model = fastdeploy::vision::tracking::PPTracking(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  fastdeploy::vision::MOTResult result;
  fastdeploy::vision::tracking::TrailRecorder trail_recorder;
  // during each prediction, data is inserted into the recorder. As the number of predictions increases,
  // the memory will continue to grow. You can cancel the insertion through 'UnbindRecorder'.
  // int count = 0; // unbind condition
  model.BindRecorder(&trail_recorder);
  cv::Mat frame;
  cv::VideoCapture capture(video_file);
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    if (!model.Predict(&frame, &result)) {
      std::cerr << "Failed to predict." << std::endl;
      return;
    }
    // such as adding this code can cancel trail datat bind
    //if(count++ == 10) model.UnbindRecorder();
    // std::cout << result.Str() << std::endl;
    cv::Mat out_img = fastdeploy::vision::VisMOT(frame, result, 0.0, &trail_recorder);
    cv::imshow("mot",out_img);
    cv::waitKey(30);
  }
  model.UnbindRecorder();
  capture.release();
  cv::destroyAllWindows();
}

void TrtInfer(const std::string& model_dir, const std::string& video_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";

  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  auto model = fastdeploy::vision::tracking::PPTracking(
      model_file, params_file, config_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  fastdeploy::vision::MOTResult result;
  fastdeploy::vision::tracking::TrailRecorder recorder;
  //during each prediction, data is inserted into the recorder. As the number of predictions increases,
  //the memory will continue to grow. You can cancel the insertion through 'UnbindRecorder'.
  // int count = 0; // unbind condition
  model.BindRecorder(&recorder);
  cv::Mat frame;
  cv::VideoCapture capture(video_file);
  while (capture.read(frame)) {
    if (frame.empty()) {
        break;
    }
    if (!model.Predict(&frame, &result)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
    }
    // such as adding this code can cancel trail datat bind
    // if(count++ == 10) model.UnbindRecorder();
    // std::cout << result.Str() << std::endl;
    cv::Mat out_img = fastdeploy::vision::VisMOT(frame, result, 0.0, &recorder);
    cv::imshow("mot",out_img);
    cv::waitKey(30);
  }
  model.UnbindRecorder();
  capture.release();
  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/video run_option, "
           "e.g ./infer_model ./pptracking_model_dir ./person.mp4 0"
        << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}
