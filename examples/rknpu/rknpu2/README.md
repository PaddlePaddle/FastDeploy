# RKNPU2模型部署

## ONNX模型转换为RKNN模型

RKNPU2不支持直接使用ONNX模型进行推导，因此我们首先需要把ONNX模型转换为RKNN模型。为了简化流程了，我新建了一个仓库负责这部分的抓换，详情请查看[LuFeng](https://github.com/Zheng-Bicheng/LuFeng)

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


