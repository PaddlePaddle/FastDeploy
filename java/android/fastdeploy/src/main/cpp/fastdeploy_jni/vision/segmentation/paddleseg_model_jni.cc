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

#include <jni.h>  // NOLINT
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/results_jni.h"  // NOLINT
#include "fastdeploy_jni/vision/segmentation/segmentation_utils_jni.h"  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_bindNative(
    JNIEnv *env, jobject thiz, jstring model_file, jstring params_file,
    jstring config_file, jobject runtime_option) {
  // TODO: implement bindNative()
  return 0;
}

JNIEXPORT jobject JNICALL
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_predictNative(
    JNIEnv *env, jobject thiz, jlong cxx_context, jobject argb8888_bitmap,
    jboolean save_image, jstring save_path, jboolean rendering, jfloat weight) {
  // TODO: implement predictNative()
  return NULL;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_vision_segmentation_PaddleSegModel_releaseNative(
    JNIEnv *env, jobject thiz, jlong cxx_context) {
  // TODO: implement releaseNative()
  return JNI_FALSE;
}

#ifdef __cplusplus
}
#endif
