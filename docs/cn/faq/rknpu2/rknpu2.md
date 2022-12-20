# RKNPU2模型部署

## 安装环境
RKNPU2模型导出只支持在x86Linux平台上进行导出，安装流程请参考[RKNPU2模型导出环境配置文档](./install_rknn_toolkit2.md)

## ONNX模型转换为RKNN模型
ONNX模型不能直接调用RK芯片中的NPU进行运算，需要把ONNX模型转换为RKNN模型，具体流程请查看[RKNPU2转换文档](./export.md)

## RKNPU2已经支持的模型列表
以下环境测试的速度均为端到端，测试环境如下:
* 设备型号: RK3588
* ARM CPU使用ONNX框架进行测试
* NPU均使用单核进行测试

| 任务场景             | 模型                | 模型版本(表示已经测试的版本)               | ARM CPU/RKNN速度(ms) |
|------------------|-------------------|-------------------------------|--------------------|
| Detection        | Picodet           | Picodet-s                     | 162/112            |
| Detection        | RKYOLOV5          | YOLOV5-S-Relu(int8)           | -/57               |
| Detection        | RKYOLOX           | -                             | -/-                |
| Detection        | RKYOLOV7          | -                             | -/-                |
| Segmentation     | Unet              | Unet-cityscapes               | -/-                |
| Segmentation     | PP-HumanSegV2Lite | portrait                      | 133/43             |
| Segmentation     | PP-HumanSegV2Lite | human                         | 133/43             |
| Face Detection   | SCRFD             | SCRFD-2.5G-kps-640            | 108/42             |

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


## 其他关联文档
- [rknpu2板端环境安装配置](../../build_and_install/rknpu2.md)
- [rknn_toolkit2安装文档](./install_rknn_toolkit2.md)
- [onnx转换rknn文档](./export.md)
