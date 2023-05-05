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

void InitAndInfer(const std::string &det_model_dir,
                  const std::string &rec_model_dir,
                  const std::string &table_model_dir,
                  const std::string &rec_label_file,
                  const std::string &table_char_dict_path,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto table_model_file = table_model_dir + sep + "inference.pdmodel";
  auto table_params_file = table_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto rec_option = option;
  auto table_option = option;

  // The rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int rec_batch_size = 1;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});
  table_option.SetTrtInputShape("x", {1, 3, 488, 488}, {1, 3, 488, 488},
                                {1, 3, 488, 488});

  // Users could save TRT cache file to disk as follow.
  det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");
  table_option.SetTrtCacheFile(table_model_dir + sep + "table_trt_cache.trt");

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);
  auto table_model = fastdeploy::vision::ocr::StructureV2Table(
      table_model_file, table_params_file, table_char_dict_path, table_option);

  assert(det_model.Initialized());
  assert(rec_model.Initialized());
  assert(table_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);

  rec_model.GetPreprocessor().SetStaticShapeInfer(true);
  rec_model.GetPreprocessor().SetRecImageShape({3, 48, 320});

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  auto ppstructurev2_table = fastdeploy::pipeline::PPStructureV2Table(
      &det_model, &rec_model, &table_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size of the rec models will be the same as the number of boxes detected
  // by the det model.
  ppstructurev2_table.SetRecBatchSize(rec_batch_size);

  if (!ppstructurev2_table.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR-Table." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!ppstructurev2_table.Predict(&im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 8) {
    std::cout << "Usage: infer_ppstructurev2_table path/to/det_model "
                 "path/to/rec_model "
                 "path/to/table_model path/to/rec_label_file "
                 "path/to/table_char_dict_path path/to/image "
                 "run_option, "
                 "e.g ./infer_ppstructurev2_table ./ch_PP-OCRv3_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, e.g. 0: run with paddle "
                 "inference on cpu;"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[7]);
  std::cout << "flag: " << flag << std::endl;

  if (flag == 0) {
    option.UseCpu();
    option.UsePaddleBackend();  // Paddle Inference
  } else if (flag == 1) {
    option.UseCpu();
    option.UseOpenVINOBackend();  // OpenVINO
  } else if (flag == 2) {
    option.UseCpu();
    option.UseOrtBackend();  // ONNX Runtime
  } else if (flag == 3) {
    option.UseCpu();
    option.UseLiteBackend();  // Paddle Lite
  } else if (flag == 4) {
    option.UseGpu();
    option.UsePaddleBackend();  // Paddle Inference
  } else if (flag == 5) {
    option.UseGpu();
    option.UsePaddleInferBackend();
    option.paddle_infer_option.collect_trt_shape = true;
    option.paddle_infer_option.enable_trt = true;  // Paddle-TensorRT
  } else if (flag == 6) {
    option.UseGpu();
    option.UseOrtBackend();  // ONNX Runtime
  } else if (flag == 7) {
    option.UseGpu();
    option.UseTrtBackend();  // TensorRT
  }

  std::string det_model_dir = argv[1];
  std::string rec_model_dir = argv[2];
  std::string table_model_dir = argv[3];
  std::string rec_label_file = argv[4];
  std::string table_char_dict_path = argv[5];
  std::string test_image = argv[6];
  InitAndInfer(det_model_dir, rec_model_dir, table_model_dir, rec_label_file,
               table_char_dict_path, test_image, option);
  return 0;
}
