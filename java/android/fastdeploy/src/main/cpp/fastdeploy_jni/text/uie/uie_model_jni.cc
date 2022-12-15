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
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/runtime_option_jni.h"  // NOLINT
#include "fastdeploy_jni/text/text_results_jni.h" // NOLINT
#ifdef ENABLE_TEXT
#include "fastdeploy/text.h"  // NOLINT
#endif

namespace fni = fastdeploy::jni;
#ifdef ENABLE_TEXT
namespace text = fastdeploy::text;
#endif

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_fastdeploy_text_uie_UIEModel_bindNative(JNIEnv *env,
                                                              jobject thiz,
                                                              jstring model_file,
                                                              jstring params_file,
                                                              jstring vocab_file,
                                                              jfloat position_prob,
                                                              jint max_length,
                                                              jobjectArray schema,
                                                              jobject runtime_option,
                                                              jint schema_language) {
#ifndef ENABLE_TEXT
  return 0;
#else
  auto c_model_file = fni::ConvertTo<std::string>(env, model_file);
  auto c_params_file = fni::ConvertTo<std::string>(env, params_file);
  auto c_vocab_file = fni::ConvertTo<std::string>(env, vocab_file);
  auto c_position_prob = static_cast<jfloat>(position_prob);
  auto c_max_length = static_cast<size_t>(max_length);
  auto c_schema = fni::ConvertTo<std::vector<std::string>>(env, schema);
  auto c_runtime_option = fni::NewCxxRuntimeOption(env, runtime_option);

  auto c_model_ptr = new text::UIEModel(c_model_file,
                                        c_params_file,
                                        c_vocab_file,
                                        c_position_prob,
                                        c_max_length,
                                        c_schema,
                                        c_runtime_option);
  INITIALIZED_OR_RETURN(c_model_ptr)

#ifdef ENABLE_RUNTIME_PERF
  c_model_ptr->EnableRecordTimeOfRuntime();
#endif
  return reinterpret_cast<jlong>(c_model_ptr);
#endif
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_fastdeploy_text_uie_UIEModel_predictNative(JNIEnv *env,
                                                                 jobject thiz,
                                                                 jlong cxx_context,
                                                                 jobjectArray texts) {
#ifndef ENABLE_TEXT
  return NULL;
#else
  if (cxx_context == 0) {
    return NULL;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel *>(cxx_context);
  auto c_texts = fni::ConvertTo<std::vector<std::string>>(env, texts);
  std::vector<std::unordered_map<std::string, std::vector<text::UIEResult>>> results;
  auto t = fni::GetCurrentTime();
  c_model_ptr->Predict(c_texts, &results);
  PERF_TIME_OF_RUNTIME(c_model_ptr, t)
  if (results.empty()) {
    return NULL;
  }
  // Push to HashMap Array
  const jclass j_hashmap_clazz = env->FindClass("java/util/HashMap");
  const jmethodID j_hashmap_init = env->GetMethodID(
      j_hashmap_clazz, "<init>", "()V");
  const jmethodID j_hashmap_put = env->GetMethodID(
      j_hashmap_clazz,"put",
      "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
  const int c_uie_result_size = static_cast<int>(results.size());
  jobjectArray j_hashmap_uie_result_arr = env->NewObjectArray(
      c_uie_result_size, j_hashmap_clazz, NULL);

  const jclass j_uie_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/text/UIEResult");

  for (int i = 0; i < c_uie_result_size; ++i) {
    auto& curr_uie_result_map = results[i];

    // From: std::unordered_map<std::string, std::vector<text::UIEResult>>
    //   To: HashMap<String, UIEResult[]>
    jobject curr_j_uie_result_hashmap = env->NewObject(j_hashmap_clazz, j_hashmap_init);

    for (auto&& curr_uie_result: curr_uie_result_map) {
      const auto& curr_uie_key = curr_uie_result.first;
      jstring curr_inner_j_uie_key = fni::ConvertTo<jstring>(env, curr_uie_key);
      // Convert std::vector<UIEResult> -> Java UIEResult[]
      const int curr_inner_c_uie_result_num = curr_uie_result.second.size();
      if (curr_inner_c_uie_result_num > 0) {
        jobjectArray curr_inner_j_uie_result_arr = env->NewObjectArray(
            curr_inner_c_uie_result_num, j_uie_result_clazz, NULL);
        for (int j = 0; j < curr_inner_c_uie_result_num; ++j) {
          text::UIEResult* inner_cxx_uie_result = (&(curr_uie_result.second[j]));
          jobject curr_inner_j_uie_result_obj = fni::NewUIEJavaResultFromCxx(
              env, reinterpret_cast<void *>(inner_cxx_uie_result));
          env->SetObjectArrayElement(curr_inner_j_uie_result_arr, j,
                                     curr_inner_j_uie_result_obj);
          env->DeleteLocalRef(curr_inner_j_uie_result_obj);
        }
        // Set element of 'curr_j_uie_result_hashmap': HashMap<String, UIEResult[]>
        env->CallObjectMethod(curr_j_uie_result_hashmap, j_hashmap_put, curr_inner_j_uie_key,
                              curr_inner_j_uie_result_arr);

        env->DeleteLocalRef(curr_inner_j_uie_key);
        env->DeleteLocalRef(curr_inner_j_uie_result_arr);
      }
    }
    // Set current HashMap<String, UIEResult[]> to HashMap[]
    env->SetObjectArrayElement(j_hashmap_uie_result_arr, i, curr_j_uie_result_hashmap);
    env->DeleteLocalRef(curr_j_uie_result_hashmap);
  }

  return j_hashmap_uie_result_arr;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_text_uie_UIEModel_releaseNative(JNIEnv *env,
                                                                 jobject thiz,
                                                                 jlong cxx_context) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  if (cxx_context == 0) {
    return JNI_FALSE;
  }
  auto c_model_ptr = reinterpret_cast<text::UIEModel *>(cxx_context);
  PERF_TIME_OF_RUNTIME(c_model_ptr, -1)

  delete c_model_ptr;
  LOGD("[End] Release UIEModel in native !");
  return JNI_TRUE;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_text_uie_UIEModel_setSchemaStringNative(
    JNIEnv *env, jobject thiz, jobjectArray schema) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  // TODO: implement setSchemaFromStringNative()
  return JNI_TRUE;
#endif
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_fastdeploy_text_uie_UIEModel_setSchemaNodeNative(
    JNIEnv *env, jobject thiz, jobjectArray schema) {
#ifndef ENABLE_TEXT
  return JNI_FALSE;
#else
  // TODO: implement setSchemaFromSchemaNodeNative()
  return JNI_TRUE;
#endif
}

#ifdef __cplusplus
}
#endif

