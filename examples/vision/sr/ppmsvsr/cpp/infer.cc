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

void CpuInfer(const std::string& model_dir, const std::string& video_file,
              int frame_num) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto model = fastdeploy::vision::sr::PPMSVSR(model_file, params_file);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  // note: input/output shape is [b, n, c, h, w] (n = frame_nums; b=1(default))
  // b and n is dependent on export model shape
  // see
  // https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
  cv::VideoCapture capture;
  // change your save video path
  std::string video_out_name = "output.mp4";
  capture.open(video_file);
  if (!capture.isOpened()) {
    std::cout << "can not open video " << std::endl;
    return;
  }
  // Get Video info :fps, frame count
  // it used 4.x version of opencv below
  // notice your opencv version and method of api.
  int video_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
  int video_frame_count =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  // Set fixed size for output frame, only for msvsr model
  int out_width = 1280;
  int out_height = 720;
  std::cout << "fps: " << video_fps << "\tframe_count: " << video_frame_count
            << std::endl;

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path("./");
  video_out_path += video_out_name;
  int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  video_out.open(video_out_path, fcc, video_fps,
                 cv::Size(out_width, out_height), true);
  if (!video_out.isOpened()) {
    std::cout << "create video writer failed!" << std::endl;
    return;
  }
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  bool reach_end = false;
  while (capture.isOpened()) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < frame_num; i++) {
      capture.read(frame);
      if (!frame.empty()) {
        imgs.push_back(frame);
      } else {
        reach_end = true;
      }
    }
    if (reach_end) {
      break;
    }
    std::vector<cv::Mat> results;
    model.Predict(imgs, results);
    for (auto& item : results) {
      // cv::imshow("13",item);
      // cv::waitKey(30);
      video_out.write(item);
      std::cout << "Processing frame: " << frame_id << std::endl;
      frame_id += 1;
    }
  }
  std::cout << "inference finished, output video saved at " << video_out_path
            << std::endl;
  capture.release();
  video_out.release();
}

void GpuInfer(const std::string& model_dir, const std::string& video_file,
              int frame_num) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";

  auto option = fastdeploy::RuntimeOption();
  // use paddle-TRT
  option.UseGpu();
  auto model = fastdeploy::vision::sr::PPMSVSR(model_file, params_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  // note: input/output shape is [b, n, c, h, w] (n = frame_nums; b=1(default))
  // b and n is dependent on export model shape
  // see
  // https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
  cv::VideoCapture capture;
  // change your save video path
  std::string video_out_name = "output.mp4";
  capture.open(video_file);
  if (!capture.isOpened()) {
    std::cout << "can not open video " << std::endl;
    return;
  }
  // Get Video info :fps, frame count
  int video_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
  int video_frame_count =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  // Set fixed size for output frame, only for msvsr model
  int out_width = 1280;
  int out_height = 720;
  std::cout << "fps: " << video_fps << "\tframe_count: " << video_frame_count
            << std::endl;

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path("./");
  video_out_path += video_out_name;
  int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  video_out.open(video_out_path, fcc, video_fps,
                 cv::Size(out_width, out_height), true);
  if (!video_out.isOpened()) {
    std::cout << "create video writer failed!" << std::endl;
    return;
  }
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  bool reach_end = false;
  while (capture.isOpened()) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < frame_num; i++) {
      capture.read(frame);
      if (!frame.empty()) {
        imgs.push_back(frame);
      } else {
        reach_end = true;
      }
    }
    if (reach_end) {
      break;
    }
    std::vector<cv::Mat> results;
    model.Predict(imgs, results);
    for (auto& item : results) {
      // cv::imshow("13",item);
      // cv::waitKey(30);
      video_out.write(item);
      std::cout << "Processing frame: " << frame_id << std::endl;
      frame_id += 1;
    }
  }
  std::cout << "inference finished, output video saved at " << video_out_path
            << std::endl;
  capture.release();
  video_out.release();
}

void TrtInfer(const std::string& model_dir, const std::string& video_file,
              int frame_num) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  option.EnablePaddleTrtCollectShape();
  option.SetTrtInputShape("lqs", {1, 2, 3, 180, 320});
  option.EnablePaddleToTrt();
  auto model = fastdeploy::vision::sr::PPMSVSR(model_file, params_file, option);

  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  // note: input/output shape is [b, n, c, h, w] (n = frame_nums; b=1(default))
  // b and n is dependent on export model shape
  // see
  // https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md
  cv::VideoCapture capture;
  // change your save video path
  std::string video_out_name = "output.mp4";
  capture.open(video_file);
  if (!capture.isOpened()) {
    std::cout << "can not open video " << std::endl;
    return;
  }
  // Get Video info :fps, frame count
  int video_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
  int video_frame_count =
      static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  // Set fixed size for output frame, only for msvsr model
  // Note that the resolution between the size and the original input is
  // consistent when the model is exported,
  // for example: [1,2,3,180,320], after 4x super separation [1,2,3,720,1080].
  // Therefore, it is very important to derive the model
  int out_width = 1280;
  int out_height = 720;
  std::cout << "fps: " << video_fps << "\tframe_count: " << video_frame_count
            << std::endl;

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path("./");
  video_out_path += video_out_name;
  int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  video_out.open(video_out_path, fcc, video_fps,
                 cv::Size(out_width, out_height), true);
  if (!video_out.isOpened()) {
    std::cout << "create video writer failed!" << std::endl;
    return;
  }
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  bool reach_end = false;
  while (capture.isOpened()) {
    std::vector<cv::Mat> imgs;
    for (int i = 0; i < frame_num; i++) {
      capture.read(frame);
      if (!frame.empty()) {
        imgs.push_back(frame);
      } else {
        reach_end = true;
      }
    }
    if (reach_end) {
      break;
    }
    std::vector<cv::Mat> results;
    model.Predict(imgs, results);
    for (auto& item : results) {
      // cv::imshow("13",item);
      // cv::waitKey(30);
      video_out.write(item);
      std::cout << "Processing frame: " << frame_id << std::endl;
      frame_id += 1;
    }
  }
  std::cout << "inference finished, output video saved at " << video_out_path
            << std::endl;
  capture.release();
  video_out.release();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/model_dir path/to/video frame "
                 "number run_option, "
                 "e.g ./infer_model ./vsr_model_dir ./vsr_src.mp4 0 2"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  int frame_num = 2;
  if (argc == 5) {
    frame_num = std::atoi(argv[4]);
  }
  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2], frame_num);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2], frame_num);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2], frame_num);
  }
  return 0;
}
