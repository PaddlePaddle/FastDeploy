English | [中文](../../../cn/faq/rknpu2/rknpu2.md) 
# RKNPU2 Model Deployment

## Installation Environment
RKNPU2 model export is only supported on x86 Linux platform, please refer to [RKNPU2 Model Export Environment Configuration](./install_rknn_toolkit2.md).

## Convert ONNX to RKNN
Since the ONNX model cannot directly calculate by calling the NPU, it is necessary to convert the ONNX model to RKNN model. For detailed information, please refer to [RKNPU2 Conversion Document](./export.md).

## Models supported for RKNPU2
The following tests are at end-to-end speed, and the test environment is as follows:
* Device Model: RK3588
* ARM CPU is tested on ONNX
* with single-core NPU


| Mission Scenario               | Model                | Model Version(tested version)               | ARM CPU/RKNN speed(ms) |
|------------------|-------------------|-------------------------------|--------------------|
| Detection        | Picodet           | Picodet-s                     | 162/112            |
| Detection        | RKYOLOV5          | YOLOV5-S-Relu(int8)           | -/57               |
| Detection        | RKYOLOX           | -                             | -/-                |
| Detection        | RKYOLOV7          | -                             | -/-                |
| Segmentation     | Unet              | Unet-cityscapes               | -/-                |
| Segmentation     | PP-HumanSegV2Lite | portrait                      | 133/43             |
| Segmentation     | PP-HumanSegV2Lite | human                         | 133/43             |
| Face Detection   | SCRFD             | SCRFD-2.5G-kps-640            | 108/42             |


## How to use RKNPU2 Backend to Infer Models

We provide an example on Scrfd model here to show how to use RKNPU2 Backend for model inference. The modifications mentioned in the annotations below are in comparison to the ONNX CPU.

```c++
int infer_scrfd_npu() {
    char model_path[] = "./model/scrfd_2.5g_bnkps_shape640x640.rknn";
    char image_file[] = "./image/test_lite_face_detector_3.jpg";
    auto option = fastdeploy::RuntimeOption();
	// Modification1: option.UseRKNPU2 function should be called
    option.UseRKNPU2();  

	// Modification2: The parameter 'fastdeploy::ModelFormat::RKNN' should be transferred when loading the model
    auto *model = new fastdeploy::vision::facedet::SCRFD(model_path,"",option,fastdeploy::ModelFormat::RKNN);  
    if (!model->Initialized()) {
        std::cerr << "Failed to initialize." << std::endl;
        return 0;
    }

	// Modification3(optional): RKNPU2 supports to normalize using NPU and the input format is nhwc format.
	// The action of DisableNormalizeAndPermute will block the nor action and hwc to chw converting action during preprocessing.
	// If you use an already supported model list, please call its method before Predict.
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


## Other related Documents
- [How to Build RKNPU2 Deployment Environment](../../build_and_install/rknpu2.md)
- [RKNN-Toolkit2 Installation Document](./install_rknn_toolkit2.md)
- [How to convert ONNX to RKNN](./export.md)
