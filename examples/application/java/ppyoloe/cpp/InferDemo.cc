#include "InferDemo.h"

#include "fastdeploy/vision.h"

std::string ConvertTo(JNIEnv *env, jstring jstr) {
  if (!jstr) {
    return "";
  }
  const jclass jstring_clazz = env->GetObjectClass(jstr);
  const jmethodID getBytesID =
      env->GetMethodID(jstring_clazz, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray jstring_bytes = reinterpret_cast<jbyteArray>(
      env->CallObjectMethod(jstr, getBytesID, env->NewStringUTF("UTF-8")));

  size_t length = static_cast<size_t>(env->GetArrayLength(jstring_bytes));
  jbyte *jstring_bytes_ptr = env->GetByteArrayElements(jstring_bytes, NULL);

  std::string res =
      std::string(reinterpret_cast<char *>(jstring_bytes_ptr), length);
  env->ReleaseByteArrayElements(jstring_bytes, jstring_bytes_ptr, JNI_ABORT);

  env->DeleteLocalRef(jstring_bytes);
  env->DeleteLocalRef(jstring_clazz);
  return res;
}

JNIEXPORT void JNICALL Java_InferDemo_infer(JNIEnv *env, jobject thiz,
                                            jstring modelPath,
                                            jstring imagePath) {
  std::string model_path = ConvertTo(env, modelPath);
  if (model_path[model_path.length() - 1] != '/') {
    model_path += "/";
  }
  std::string model_file = model_path + "model.pdmodel";
  std::string params_file = model_path + "model.pdiparams";
  std::string infer_cfg_file = model_path + "infer_cfg.yml";

  // 模型推理的配置信息
  fastdeploy::RuntimeOption option;
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file,
                                                      infer_cfg_file, option);

  assert(model.Initialized());  // 判断模型是否初始化成功

  std::string image_path = ConvertTo(env, imagePath);
  cv::Mat im = cv::imread(image_path);
  fastdeploy::vision::DetectionResult result;

  assert(model.Predict(&im, &result));  // 判断是否预测成功

  std::cout << result.Str() << std::endl;

  cv::Mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  // 可视化结果保存到本地
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result save in vis_result.jpg" << std::endl;
}