//
// Created by aichao on 2022/10/10.
//

#include "fastdeploy/vision/tracking/pptracking/predictor.h"
namespace fastdeploy {
namespace vision {
namespace tracking {

PPTracking::PPTracking(const std::string& model_file, const std::string& params_file,
               const RuntimeOption& custom_option,
               const ModelFormat& model_format){
  if (model_format == ModelFormat::ONNX)
  {
    valid_cpu_backends = {Backend::ORT,Backend::OPENVINO};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::OPENVINO};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  std::cout<<"--------"<<std::endl;

  initialized = Initialize();
}

bool PPTracking::Initialize() {
  target_size_ ={320,576};
  mean_={0.0f,0.0f,0.0f};
  scale_={1.0f,1.0f,1.0f};
  conf_thresh_=0.4;
    threshold_=0.5;
    min_box_area_=200;

  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool PPTracking::Predict(cv::Mat *img, MOTResult *result) {
  Mat mat(*img);
  std::vector<FDTensor> input_tensors;

  if (!Preprocess(&mat, &input_tensors)) {
    FDERROR << "Failed to preprocess input image." << std::endl;
    return false;
  }
  float* tmp = static_cast<float*>(input_tensors[1].Data());

  std::vector<FDTensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    FDERROR << "Failed to inference." << std::endl;
    return false;
  }

  if (!Postprocess(output_tensors, result)) {
    FDERROR << "Failed to post process." << std::endl;
    return false;
  }

  return true;


}

void PPTracking::LetterBoxResize(Mat* im){

    // generate scale_factor
    int origin_w = im->Width();
    int origin_h = im->Height();
    int target_h = target_size_[0];
    int target_w = target_size_[1];
    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);

    int new_shape_w = std::round(im->Width() * resize_scale);
    int new_shape_h = std::round(im->Height() * resize_scale);
    float padw = (target_size_[1] - new_shape_w) / 2.;
    float padh = (target_size_[0] - new_shape_h) / 2.;
    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    Resize::Run(im,new_shape_w,new_shape_h);
    std::vector<float> color{127.5,127.5,127.5};
    Pad::Run(im,top,bottom,left,right,color);
}

bool PPTracking::Preprocess(Mat* mat, std::vector<FDTensor>* outputs) {

    int origin_w = mat->Width();
    int origin_h = mat->Height();
    LetterBoxResize(mat);
    Normalize::Run(mat,mean_,scale_,is_scale_);
    HWC2CHW::Run(mat);
    Cast::Run(mat, "float");

    outputs->resize(3);
    // image_shape
    (*outputs)[0].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(0).name);
    float* shape = static_cast<float*>((*outputs)[0].MutableData());
    shape[0] = mat->Height();
    shape[1] = mat->Width();
    // image
    (*outputs)[1].name = InputInfoOfRuntime(1).name;
    mat->ShareWithTensor(&((*outputs)[1]));
    (*outputs)[1].ExpandDim(0);
    // scale
    (*outputs)[2].Allocate({1, 2}, FDDataType::FP32, InputInfoOfRuntime(2).name);
    float* scale = static_cast<float*>((*outputs)[2].MutableData());
    scale[0] = mat->Height() * 1.0 / origin_h;
    scale[1] = mat->Width() * 1.0 / origin_w;
    return true;
}


void FilterDets(const float conf_thresh,const cv::Mat dets,std::vector<int>* index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
      if (score > conf_thresh) {
        index->push_back(i);
      }
  }
}

bool PPTracking::Postprocess(std::vector<FDTensor>& infer_result, MOTResult *result){


    auto bbox_shape = infer_result[0].shape;
    auto bbox_data = static_cast<float*>(infer_result[0].Data());

    auto emb_shape = infer_result[1].shape;
    auto emb_data = static_cast<float*>(infer_result[1].Data());

    cv::Mat dets(bbox_shape[0], 6, CV_32FC1, bbox_data);
    cv::Mat emb(bbox_shape[0], emb_shape[1], CV_32FC1, emb_data);


    result->clear();
    std::vector<Track> tracks;
    std::vector<int> valid;
    FilterDets(conf_thresh_, dets, &valid);
    cv::Mat new_dets, new_emb;
    for (int i = 0; i < valid.size(); ++i) {
        new_dets.push_back(dets.row(valid[i]));
        new_emb.push_back(emb.row(valid[i]));
    }
    JDETracker::instance()->update(new_dets, new_emb, &tracks);
    if (tracks.size() == 0) {
        MOTTrack mot_track;
        Rect ret = {*dets.ptr<float>(0, 0),
                    *dets.ptr<float>(0, 1),
                    *dets.ptr<float>(0, 2),
                    *dets.ptr<float>(0, 3)};
        mot_track.ids = 1;
        mot_track.score = *dets.ptr<float>(0, 4);
        mot_track.rects = ret;
        result->push_back(mot_track);
    } else {
        std::vector<Track>::iterator titer;
        for (titer = tracks.begin(); titer != tracks.end(); ++titer) {
            if (titer->score < threshold_) {
                continue;
            } else {
                float w = titer->ltrb[2] - titer->ltrb[0];
                float h = titer->ltrb[3] - titer->ltrb[1];
                bool vertical = w / h > 1.6;
                float area = w * h;
                if (area > min_box_area_ && !vertical) {
                    MOTTrack mot_track;
                    Rect ret = {
                            titer->ltrb[0], titer->ltrb[1], titer->ltrb[2], titer->ltrb[3]};
                    mot_track.rects = ret;
                    mot_track.score = titer->score;
                    mot_track.ids = titer->id;
                    result->push_back(mot_track);
                }
            }
        }
    }
    return true;
}

cv::Mat PPTracking::Visualize(const cv::Mat& img,
                                 const MOTResult& results,
                                 const float fps,
                                 const int frame_id){

return VisualizeTrackResult(img,results,fps,frame_id);

}

} // namespace tracking
} // namespace vision
} // namespace fastdeploy
