#include "fastdeploy/vision/ultralytics/yolov5.h"
#include "fastdeploy/utils/perf.h"
#include "fastdeploy/vision/utils/utils.h"

namespace fastdeploy {
namespace vision {
namespace ultralytics {

void LetterBox(Mat* mat, std::vector<int> size, std::vector<float> color,
               bool _auto, bool scale_fill = false, bool scale_up = true,
               int stride = 32) {
  float scale =
      std::min(size[1] * 1.0 / mat->Height(), size[0] * 1.0 / mat->Width());
  if (!scale_up) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
  if (_auto) {
    pad_h = pad_h % stride;
    pad_w = pad_w % stride;
  } else if (scale_fill) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size[1];
    resize_w = size[0];
  }
  Resize::Run(mat, resize_w, resize_h);
  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, color);
  }
}

YOLOv5::YOLOv5(const std::string& model_file,
               const RuntimeOption& custom_option,
               const Frontend& model_format) {
  valid_cpu_backends = {Backend::ORT}; // 指定可用的CPU后端
  valid_gpu_backends = {Backend::ORT}; // 指定可用的GPU后端
  runtime_option = custom_option;
  runtime_option.model_format = model_format; // 指定模型格式
  runtime_option.model_file = model_file;
  // initialized用于标记模型是否初始化成功
  // C++或Python中可调用YOLOv5.Intialized() /
  // YOLOv5.initialized()判断模型是否初始化成功
  initialized = Initialize();
}

YOLOv5::YOLOv5(const std::string& model_file, const std::string& params_file,
               const RuntimeOption& custom_option,
               const Frontend& model_format) {
  valid_cpu_backends = {Backend::PDRT}; // 指定可用的CPU后端
  valid_gpu_backends = {Backend::PDRT}; // 指定可用的GPU后端
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
  initialized = Initialize();
}

bool YOLOv5::Initialize() {
  // parameters for preprocess
  size = {640, 640};
  padding_value = {114.0, 114.0, 114.0};
  is_mini_pad = false;
  is_no_pad = false;
  is_scale_up = true;
  stride = 32;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool YOLOv5::Preprocess(Mat* mat, FDTensor* output,
                        std::map<std::string, std::array<float, 2>>* im_info) {
  // yolov5's preprocess steps
  // 1. letterbox
  // 2. BGR->RGB
  // 3. HWC->CHW
  LetterBox(mat, size, padding_value, is_mini_pad, is_no_pad, is_scale_up,
            stride);
  BGR2RGB::Run(mat);
  Normalize::Run(mat, std::vector<float>(mat->Channels(), 0.0),
                 std::vector<float>(mat->Channels(), 1.0));

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1); // reshape to n, h, w, c
  return true;
}

bool YOLOv5::Postprocess(
    FDTensor& infer_result, DetectionResult* result,
    const std::map<std::string, std::array<float, 2>>& im_info,
    float conf_threshold, float nms_iou_threshold) {
  FDASSERT(infer_result.shape[0] == 1, "Only support batch =1 now.");
  result->Clear();
  result->Reserve(infer_result.shape[1]);
  if (infer_result.dtype != FDDataType::FP32) {
    FDERROR << "Only support post process with float32 data." << std::endl;
    return false;
  }
  float* data = static_cast<float*>(infer_result.Data());
  for (size_t i = 0; i < infer_result.shape[1]; ++i) {
    int s = i * infer_result.shape[2];
    float confidence = data[s + 4];
    float* max_class_score =
        std::max_element(data + s + 5, data + s + infer_result.shape[2]);
    confidence *= (*max_class_score);
    // filter boxes by conf_threshold
    if (confidence <= conf_threshold) {
      continue;
    }
    // convert from [x, y, w, h] to [x1, y1, x2, y2]
    result->boxes.emplace_back(std::array<float, 4>{
        data[s] - data[s + 2] / 2, data[s + 1] - data[s + 3] / 2,
        data[s + 0] + data[s + 2] / 2, data[s + 1] + data[s + 3] / 2});
    result->label_ids.push_back(std::distance(data + s + 5, max_class_score));
    result->scores.push_back(confidence);
  }
  utils::NMS(result, nms_iou_threshold);

  // scale the boxes to the origin image shape
  auto iter_out = im_info.find("output_shape");
  auto iter_ipt = im_info.find("input_shape");
  FDASSERT(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Cannot find input_shape or output_shape from im_info.");
  float out_h = iter_out->second[0];
  float out_w = iter_out->second[1];
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];
  float scale = std::min(out_h / ipt_h, out_w / ipt_w);
  for (size_t i = 0; i < result->boxes.size(); ++i) {
    float pad_h = (out_h - ipt_h * scale) / 2;
    float pad_w = (out_w - ipt_w * scale) / 2;

    // clip box
    result->boxes[i][0] = std::max((result->boxes[i][0] - pad_w) / scale, 0.0f);
    result->boxes[i][1] = std::max((result->boxes[i][1] - pad_h) / scale, 0.0f);
    result->boxes[i][2] = std::max((result->boxes[i][2] - pad_w) / scale, 0.0f);
    result->boxes[i][3] = std::max((result->boxes[i][3] - pad_h) / scale, 0.0f);
    result->boxes[i][0] = std::min(result->boxes[i][0], ipt_w);
    result->boxes[i][1] = std::min(result->boxes[i][1], ipt_h);
    result->boxes[i][2] = std::min(result->boxes[i][2], ipt_w);
    result->boxes[i][3] = std::min(result->boxes[i][3], ipt_h);
  }
  return true;
}

bool YOLOv5::Predict(cv::Mat* im, DetectionResult* result, float conf_threshold,
                     float nms_iou_threshold) {

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_START(0)
#endif

  Mat mat(*im);
  std::vector<FDTensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(0, "Preprocess")
  TIMERECORD_START(1)
#endif

  input_tensors[0].name = InputInfoOfRuntime(0).name;

  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(1, "Inference")
  TIMERECORD_START(2)
#endif

  if (!Postprocess(output_tensors[0], result, im_info, conf_threshold,
                   nms_iou_threshold)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

#ifdef FASTDEPLOY_DEBUG
  TIMERECORD_END(2, "Postprocess")
#endif
  return true;
}

} // namespace ultralytics
} // namespace vision
} // namespace fastdeploy
