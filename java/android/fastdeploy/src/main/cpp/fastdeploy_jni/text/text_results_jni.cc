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

#include <jni.h>             // NOLINT
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/convert_jni.h"  // NOLINT
#include "fastdeploy_jni/text/text_results_jni.h" // NOLINT

namespace fastdeploy {
namespace jni {

jobject NewUIEJavaResultFromCxx(JNIEnv *env, const text::UIEResult& cxx_uie_result) {
  // Field signatures of Java UIEResult:
  // (1) mStart long:                              J
  // (2) mEnd long:                                J
  // (3) mProbability double:                      D
  // (4) mText String:                             Ljava/lang/String;
  // (5) mRelation HashMap<String, UIEResult[]>:   Ljava/util/HashMap;
  // (6) mInitialized boolean:                     Z
  const int len = static_cast<int>(cxx_uie_result.text_.size());
  if (len == 0) {
    return NULL;
  }
  const jclass j_uie_result_clazz = env->FindClass(
      "com/baidu/paddle/fastdeploy/text/UIEResult");
  const jfieldID j_uie_start_id = env->GetFieldID(
      j_uie_result_clazz, "mStart", "J");
  const jfieldID j_uie_end_id = env->GetFieldID(
      j_uie_result_clazz, "mEnd", "J");
  const jfieldID j_uie_probability_id = env->GetFieldID(
      j_uie_result_clazz, "mProbability", "D");
  const jfieldID j_uie_text_id = env->GetFieldID(
      j_uie_result_clazz, "mText", "Ljava/lang/String;");
  const jfieldID j_uie_relation_id = env->GetFieldID(
      j_uie_result_clazz, "mRelation", "Ljava/util/HashMap;");
  const jfieldID j_uie_initialized_id = env->GetFieldID(
      j_uie_result_clazz, "mInitialized", "Z");
  // Default UIEResult constructor.
  const jmethodID j_uie_result_init = env->GetMethodID(
      j_uie_result_clazz, "<init>", "()V");

  jobject j_uie_result_obj = env->NewObject(j_uie_result_clazz, j_uie_result_init);

  // Allocate for current UIEResult
  // mStart long: J & mEnd   long: J
  env->SetLongField(j_uie_result_obj, j_uie_start_id,
                    static_cast<jlong>(cxx_uie_result.start_));
  env->SetLongField(j_uie_result_obj, j_uie_end_id,
                    static_cast<jlong>(cxx_uie_result.end_));
  // mProbability double: D
  env->SetDoubleField(j_uie_result_obj, j_uie_probability_id,
                      static_cast<jdouble>(cxx_uie_result.probability_));
  // mText String: Ljava/lang/String;
  env->SetObjectField(j_uie_result_obj, j_uie_text_id,
                      ConvertTo<jstring>(env, cxx_uie_result.text_));
  // mInitialized boolean: Z
  env->SetBooleanField(j_uie_result_obj, j_uie_initialized_id, JNI_TRUE);

  // mRelation HashMap<String, UIEResult[]>: Ljava/util/HashMap;
  if (cxx_uie_result.relation_.size() > 0) {
    const jclass j_hashmap_clazz = env->FindClass("java/util/HashMap");
    const jmethodID j_hashmap_init = env->GetMethodID(
        j_hashmap_clazz, "<init>", "()V");
    const jmethodID j_hashmap_put = env->GetMethodID(
        j_hashmap_clazz,"put",
        "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    // std::unordered_map<std::string, std::vector<UIEResult>> relation_;
    jobject j_uie_relation_hashmap = env->NewObject(j_hashmap_clazz, j_hashmap_init);

    for (auto&& curr_relation : cxx_uie_result.relation_) {
      // Processing each key-value cxx uie relation:
      // Key: string, Value: std::vector<UIEResult>
      const auto& curr_c_relation_key = curr_relation.first;
      jstring curr_j_relation_key = ConvertTo<jstring>(env, curr_c_relation_key);
      // Init current relation array (array of UIEResult)
      const int curr_c_uie_result_size = curr_relation.second.size();
      jobjectArray curr_j_uie_result_obj_arr = env->NewObjectArray(
          curr_c_uie_result_size, j_uie_result_clazz, NULL);
      for (int i = 0; i < curr_c_uie_result_size; ++i) {
        const text::UIEResult& child_cxx_result = curr_relation.second[i];
        // Recursively generates the curr_j_uie_result_obj
        jobject curr_j_uie_result_obj = NewUIEJavaResultFromCxx(env, child_cxx_result);
        env->SetObjectArrayElement(curr_j_uie_result_obj_arr, i, curr_j_uie_result_obj);
        env->DeleteLocalRef(curr_j_uie_result_obj);
      }
      // Put current relation array (array of UIEResult) to HashMap
      env->CallObjectMethod(j_uie_relation_hashmap, j_hashmap_put, curr_j_relation_key,
                            curr_j_uie_result_obj_arr);

      env->DeleteLocalRef(curr_j_relation_key);
      env->DeleteLocalRef(curr_j_uie_result_obj_arr);
    }
    // Set relation HashMap from native
    env->SetObjectField(j_uie_result_obj, j_uie_relation_id, j_uie_relation_hashmap);
    env->DeleteLocalRef(j_hashmap_clazz);
    env->DeleteLocalRef(j_uie_relation_hashmap);
  }

  env->DeleteLocalRef(j_uie_result_clazz);

  return j_uie_result_obj;
}

}  // namespace jni
}  // namespace fastdeploy






















































