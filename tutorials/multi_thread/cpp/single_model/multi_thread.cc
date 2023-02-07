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

#include <thread>
#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void Predict(fastdeploy::vision::classification::PaddleClasModel *model, int thread_id, const std::vector<std::string>& images) {
  for (auto const &image_file : images) {
      auto im = cv::imread(image_file);

      fastdeploy::vision::ClassifyResult res;
      if (!model->Predict(im, &res)) {
        std::cerr << "Failed to predict." << std::endl;
        return;
      }

      // print res
      std::cout << "Thread Id: " << thread_id << std::endl;
      std::cout << res.Str() << std::endl;
  }
}

void GetImageList(std::vector<std::vector<std::string>>* image_list, const std::string& image_file_path, int thread_num){
  std::vector<cv::String> images;
  cv::glob(image_file_path, images, false);
  // number of image files in images folder
  size_t count = images.size(); 
  size_t num = count / thread_num;
  for (int i = 0; i < thread_num; i++) {
    std::vector<std::string> temp_list;
    if (i == thread_num - 1) {
      for (size_t j = i*num; j < count; j++){
        temp_list.push_back(images[j]);
      }
    } else {
      for (size_t j = 0; j < num; j++){
        temp_list.push_back(images[i * num + j]);
      }
    }
    (*image_list)[i] = temp_list;
  }
}

void CpuInfer(const std::string& model_dir, const std::string& image_file_path, int thread_num) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<decltype(model.Clone())> models;
  for (int i = 0; i < thread_num; ++i) {
    models.emplace_back(std::move(model.Clone()));
  }

  std::vector<std::vector<std::string>> image_list(thread_num);
  GetImageList(&image_list, image_file_path, thread_num);

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(Predict, models[i].get(), i, image_list[i]);
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

void GpuInfer(const std::string& model_dir, const std::string& image_file_path, int thread_num) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UsePaddleBackend();
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<decltype(model.Clone())> models;
  for (int i = 0; i < thread_num; ++i) {
    models.emplace_back(std::move(model.Clone()));
  }

  std::vector<std::vector<std::string>> image_list(thread_num);
  GetImageList(&image_list, image_file_path, thread_num);

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(Predict, models[i].get(), i, image_list[i]);
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

void TrtInfer(const std::string& model_dir, const std::string& image_file_path, int thread_num) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  option.UseTrtBackend();
  // for model.Clone() must SetTrtInputShape first
  option.SetTrtInputShape("inputs", {1, 3, 224, 224});
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  std::vector<decltype(model.Clone())> models;
  for (int i = 0; i < thread_num; ++i) {
    models.emplace_back(std::move(model.Clone()));
  }

  std::vector<std::vector<std::string>> image_list(thread_num);
  GetImageList(&image_list, image_file_path, thread_num);

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(Predict, models[i].get(), i, image_list[i]);
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option thread_num, "
                 "e.g ./multi_thread_demo ./ResNet50_vd ./test.jpeg 0 3"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend."
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2], std::atoi(argv[4]));
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2], std::atoi(argv[4]));
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2], std::atoi(argv[4]));
  } 
  return 0;
}

