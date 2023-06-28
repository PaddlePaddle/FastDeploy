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
#pragma once

#include "fastdeploy/core/config.h"
#ifdef ENABLE_VISION
#include "fastdeploy/vision/classification/contrib/resnet.h"
#include "fastdeploy/vision/classification/contrib/yolov5cls/yolov5cls.h"
#include "fastdeploy/vision/classification/ppcls/model.h"
#include "fastdeploy/vision/classification/ppshitu/ppshituv2_rec.h"
#include "fastdeploy/vision/classification/ppshitu/ppshituv2_det.h"
#include "fastdeploy/vision/detection/contrib/nanodet_plus.h"
#include "fastdeploy/vision/detection/contrib/scaledyolov4.h"
#include "fastdeploy/vision/detection/contrib/yolor.h"
#include "fastdeploy/vision/detection/contrib/yolov5/yolov5.h"
#include "fastdeploy/vision/detection/contrib/yolov5seg/yolov5seg.h"
#include "fastdeploy/vision/detection/contrib/fastestdet/fastestdet.h"
#include "fastdeploy/vision/detection/contrib/yolov5lite.h"
#include "fastdeploy/vision/detection/contrib/yolov6.h"
#include "fastdeploy/vision/detection/contrib/yolov7/yolov7.h"
#include "fastdeploy/vision/detection/contrib/yolov7end2end_ort.h"
#include "fastdeploy/vision/detection/contrib/yolov7end2end_trt.h"
#include "fastdeploy/vision/detection/contrib/yolov8/yolov8.h"
#include "fastdeploy/vision/detection/contrib/yolox.h"
#include "fastdeploy/vision/detection/contrib/rknpu2/model.h"
#include "fastdeploy/vision/perception/paddle3d/smoke/smoke.h"
#include "fastdeploy/vision/perception/paddle3d/petr/petr.h"
#include "fastdeploy/vision/perception/paddle3d/centerpoint/centerpoint.h"
#include "fastdeploy/vision/detection/ppdet/model.h"
#include "fastdeploy/vision/facealign/contrib/face_landmark_1000.h"
#include "fastdeploy/vision/facealign/contrib/pfld.h"
#include "fastdeploy/vision/facealign/contrib/pipnet.h"
#include "fastdeploy/vision/facedet/contrib/retinaface.h"
#include "fastdeploy/vision/facedet/contrib/scrfd.h"
#include "fastdeploy/vision/facedet/contrib/ultraface.h"
#include "fastdeploy/vision/facedet/contrib/yolov5face.h"
#include "fastdeploy/vision/facedet/contrib/yolov7face/yolov7face.h"
#include "fastdeploy/vision/facedet/contrib/centerface/centerface.h"
#include "fastdeploy/vision/facedet/ppdet/blazeface/blazeface.h"
#include "fastdeploy/vision/faceid/contrib/insightface/model.h"
#include "fastdeploy/vision/faceid/contrib/adaface/adaface.h"
#include "fastdeploy/vision/headpose/contrib/fsanet.h"
#include "fastdeploy/vision/keypointdet/pptinypose/pptinypose.h"
#include "fastdeploy/vision/matting/contrib/modnet.h"
#include "fastdeploy/vision/matting/contrib/rvm.h"
#include "fastdeploy/vision/matting/ppmatting/ppmatting.h"
#include "fastdeploy/vision/ocr/ppocr/classifier.h"
#include "fastdeploy/vision/ocr/ppocr/dbdetector.h"
#include "fastdeploy/vision/ocr/ppocr/structurev2_table.h"
#include "fastdeploy/vision/ocr/ppocr/structurev2_layout.h"
#include "fastdeploy/vision/ocr/ppocr/structurev2_ser_vi_layoutxlm.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v2.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v3.h"
#include "fastdeploy/vision/ocr/ppocr/ppocr_v4.h"
#include "fastdeploy/vision/ocr/ppocr/ppstructurev2_table.h"
#include "fastdeploy/vision/ocr/ppocr/ppstructurev2_layout.h"
#include "fastdeploy/vision/ocr/ppocr/recognizer.h"
#include "fastdeploy/vision/ocr/ppocr/utils/ocr_utils.h"
#include "fastdeploy/vision/segmentation/ppseg/model.h"
#include "fastdeploy/vision/sr/ppsr/model.h"
#include "fastdeploy/vision/tracking/pptracking/model.h"
#include "fastdeploy/vision/generation/contrib/animegan.h"

#endif

#include "fastdeploy/vision/visualize/visualize.h"
