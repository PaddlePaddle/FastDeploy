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
#include "fastdeploy/vision/biubug6/retinaface.h"
#include "fastdeploy/vision/deepcam/yolov5face.h"
#include "fastdeploy/vision/deepinsight/arcface.h"
#include "fastdeploy/vision/deepinsight/cosface.h"
#include "fastdeploy/vision/deepinsight/insightface_rec.h"
#include "fastdeploy/vision/deepinsight/partial_fc.h"
#include "fastdeploy/vision/deepinsight/scrfd.h"
#include "fastdeploy/vision/deepinsight/vpl.h"
#include "fastdeploy/vision/detection/contrib/nanodet_plus.h"
#include "fastdeploy/vision/detection/contrib/paddleyolox.h"
#include "fastdeploy/vision/detection/contrib/scaledyolov4.h"
#include "fastdeploy/vision/detection/contrib/yolor.h"
#include "fastdeploy/vision/detection/contrib/yolov5.h"
#include "fastdeploy/vision/detection/contrib/yolov5lite.h"
#include "fastdeploy/vision/detection/contrib/yolov6.h"
#include "fastdeploy/vision/detection/contrib/yolov7.h"
#include "fastdeploy/vision/linzaer/ultraface.h"
#include "fastdeploy/vision/ppcls/model.h"
#include "fastdeploy/vision/ppdet/model.h"
#include "fastdeploy/vision/ppseg/model.h"
#include "fastdeploy/vision/zhkkke/modnet.h"
#endif

#include "fastdeploy/vision/visualize/visualize.h"
