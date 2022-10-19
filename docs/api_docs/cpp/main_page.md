# FastDeploy C++ API Summary

## Runtime

FastDeploy Runtime can be used as an inference engine with the same code, we can deploy Paddle/ONNX model on different device by different backends.  
Currently, FastDeploy supported backends listed as below,

| Backend | Hardware | Support Model Format | Platform |
| :------ | :------- | :------------------- | :------- |
| Paddle Inference | CPU/Nvidia GPU | Paddle | Windows(x64)/Linux(x64) |
| ONNX Runtime | CPU/Nvidia GPU | Paddle/ONNX | Windows(x64)/Linux(x64/aarch64)/Mac(x86/arm64) |
| TensorRT | Nvidia GPU | Paddle/ONNX | Windows(x64)/Linux(x64)/Jetson |
| OpenVINO | CPU | Paddle/ONNX | Windows(x64)/Linux(x64)/Mac(x86) |
| Poros | CPU/Nvidia GPU | TorchScript | Linux(x64) |

### Example code
- [Python examples](./)
- [C++ examples](./)

### Related APIs
- [RuntimeOption](./structfastdeploy_1_1RuntimeOption.html)
- [Runtime](./structfastdeploy_1_1Runtime.html)

## Vision Models

| Task | Model | API | Example |
| :---- | :---- | :---- | :----- |
| object detection | PaddleDetection/PPYOLOE | [fastdeploy::vision::detection::PPYOLOE](./classfastdeploy_1_1vision_1_1detection_1_1PPYOLOE.html) | [C++](./)/[Python](./) |
| image classification | PaddleClassification serials | [fastdeploy::vision::classification::PaddleClasModel](./classfastdeploy_1_1vision_1_1classification_1_1PaddleClasModel.html) | [C++](./)/[Python](./) |
| semantic segmentation | PaddleSegmentation serials | [fastdeploy::vision::classification::PaddleSegModel](./classfastdeploy_1_1vision_1_1segmentation_1_1PaddleSegModel.html) | [C++](./)/[Python](./) |
