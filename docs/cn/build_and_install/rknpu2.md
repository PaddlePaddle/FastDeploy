# RK2代NPU部署库编译

## 写在前面
FastDeploy已经初步支持RKNPU2的部署，目前暂时仅支持c++部署。使用的过程中，如果出现Bug请提Issues反馈。

## 简介
FastDeploy当前在RK平台上支持后端引擎如下:

| 后端                | 平台                   | 支持模型格式 | 说明                                         |
|:------------------|:---------------------|:-------|:-------------------------------------------|
| ONNX&nbsp;Runtime | RK356X   <br> RK3588 | ONNX   | 编译开关`ENABLE_ORT_BACKEND`为ON或OFF控制，默认OFF    |
| RKNPU2            | RK356X   <br> RK3588 | RKNN   | 编译开关`ENABLE_RKNPU2_BACKEND`为ON或OFF控制，默认OFF |


## C++ SDK编译安装

RKNPU2仅支持linux下进行编译,以下教程均在linux环境下完成。

### 更新驱动和安装编译时需要的环境


在运行代码之前，我们需要安装以下最新的RKNPU驱动，目前驱动更新至1.4.0。为了简化安装我编写了快速安装脚本，一键即可进行安装。

**方法1: 通过脚本安装**
```bash
# 下载解压rknpu2_device_install_1.4.0
链接:https://pan.baidu.com/s/1yNww64gQnvwiCfNhELtkwQ?pwd=easy 提取码:easy 复制这段内容后打开百度网盘手机App，操作更方便哦

cd rknpu2_device_install_1.4.0
# RK3588运行以下代码
sudo rknn_install_rk3588.sh
# RK356X运行以下代码
sudo rknn_install_rk356X.sh
```

**方法2: 通过gittee安装**
```bash
# 安装必备的包
sudo apt update -y
sudo apt install -y python3 
sudo apt install -y python3-dev 
sudo apt install -y python3-pip 
sudo apt install -y gcc
sudo apt install -y python3-opencv
sudo apt install -y python3-numpy
sudo apt install -y cmake

# 下载rknpu2
# RK3588运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK3588/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK3588/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/

# RK356X运行以下代码
git clone https://gitee.com/mirrors_rockchip-linux/rknpu2.git
sudo cp ./rknpu2/runtime/RK356X/Linux/librknn_api/aarch64/* /usr/lib
sudo cp ./rknpu2/runtime/RK356X/Linux/rknn_server/aarch64/usr/bin/* /usr/bin/
```

### 编译C++ SDK

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build

# 编译配置详情见README文件，这里只介绍关键的几个配置
# -DENABLE_ORT_BACKEND:     是否开启ONNX模型，默认关闭
# -DENABLE_RKNPU2_BACKEND:  是否开启RKNPU模型，默认关闭
# -DTARGET_SOC:             编译SDK的板子型号，只能输入RK356X或者RK3588，注意区分大小写
cmake ..  -DENABLE_ORT_BACKEND=ON \
	      -DENABLE_RKNPU2_BACKEND=ON \
	      -DENABLE_VISION=ON \
	      -DTARGET_SOC=RK3588 \
          -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk
make -j8
make install
```

## RKNPU2已经支持的模型列表

| 任务场景             | 模型                | 模型版本(表示已经测试的版本)                                                                                                                            | 大小  | ONNX/RKNN是否支持 | ONNX/RKNN速度(ms) |
|------------------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----|---------------|-----------------|
| Detection        | Picodet           | [Picodet-s-npu](https://bj.bcebos.com/fastdeploy/models/rknn2/picodet_s_416_coco_npu_3588.tgz)                                             | -   | True/True     | 454/177         |
| Segmentation     | PP-LiteSeg        | [PP_LiteSeg_T_STDC1_cityscapes](https://bj.bcebos.com/fastdeploy/models/rknn2/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_3588.tgz) | -   | True/True     | 6634/5598       |
| Segmentation     | PP-HumanSegV2Lite | [portrait](https://bj.bcebos.com/fastdeploy/models/rknn2/portrait_pp_humansegv2_lite_256x144_inference_model_without_softmax_3588.tgz)     | -   | True/True     | 456/266         |
| Segmentation     | PP-HumanSegV2Lite | [human](https://bj.bcebos.com/fastdeploy/models/rknn2/human_pp_humansegv2_lite_192x192_pretrained_3588.tgz)                                | -   | True/True     | 496/256         |
| Face Detection   | SCRFD             | [SCRFD-2.5G-kps-640](https://bj.bcebos.com/fastdeploy/models/rknn2/scrfd_2.5g_bnkps_shape640x640.rknn)                                     | -   | True/True     | 963/142         |
| Face Recognition | ArcFace           | [ArcFace_r18](https://bj.bcebos.com/fastdeploy/models/rknn2/new_ms1mv3_arcface_r18.rknn)                                                   | -   | True/True     | 600/3           |
| Face Recognition | cosFace           | [cosFace_r18](https://bj.bcebos.com/fastdeploy/models/rknn2/new_glint360k_cosface_r18.rknn)                                                | -   | True/True     | 600/3           |

## RKNPU2 Backend推理使用教程

这里以Scrfd模型为例子教你如何使用RKNPU2 Backend推理模型。以下注释中的改动，是对比onnx cpu的改动。

```c++
int infer_scrfd_npu() {
    char model_path[] = "./model/scrfd_2.5g_bnkps_shape640x640.rknn";
    char image_file[] = "./image/test_lite_face_detector_3.jpg";
    auto option = fastdeploy::RuntimeOption();
	// 改动1: option需要调用UseRKNPU2
    option.UseRKNPU2();  

	// 改动2: 模型加载时需要传递fastdeploy::ModelFormat::RKNN参数
    auto *model = new fastdeploy::vision::facedet::SCRFD(model_path,"",option,fastdeploy::ModelFormat::RKNN);  
    if (!model->Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return 0;
    }

	// 改动3(可选): RKNPU2支持使用NPU进行normalize操作，并且输入格式为nhwc格式。
	// DisableNormalizeAndPermute操作将屏蔽预处理时的nor操作和hwc转chw操作。
	// 如果你使用的是已经支持的模型列表，请在Predict前调用该方法。
    model->DisableNormalizeAndPermute();
    auto im = cv::imread(image_file);
    auto im_bak = im.clone();
    fastdeploy::vision::FaceDetectionResult res;
    clock_t start = clock();
    if (!model->Predict(&im, &res, 0.8, 0.8)) {
        std::cerr << "Failed to predict." << std::endl;
        return 0;
    }
    clock_t end = clock();
    double dur = (double) (end - start);
    printf("infer_scrfd_npu use time:%f\n", (dur / CLOCKS_PER_SEC));
    auto vis_im = fastdeploy::vision::Visualize::VisFaceDetection(im_bak, res);
    cv::imwrite("scrfd_rknn_vis_result.jpg", vis_im);
    std::cout << "Visualized result saved in ./scrfd_rknn_vis_result.jpg" << std::endl;
    return 0;
}
```


