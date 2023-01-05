[简体中文](../../cn/faq/boost_cv_by_flycv.md) | English


# Accelerate end-to-end inference performance using FlyCV

[FlyCV](https://github.com/PaddlePaddle/FlyCV) is a high performance computer image processing library, providing better performance than other image processing libraries, especially in the ARM architecture.
FastDeploy is now integrated with FlyCV, allowing users to use FlyCV on supported hardware platforms to accelerate model end-to-end inference performance.

## Supported OS and Architectures

| OS | Architectures |
| :-----------| :--------   |
|   Android     |  armeabi-v7a, arm64-v8a |  
|   Linux       |  aarch64, armhf, x86_64|  


## Usage
To use FlyCV, you first need to turn on the FlyCV compile option at compile time, and then add a new line of code to turn it on.
This article uses Linux as an example to show how to enable the FlyCV compile option, and then add a new line of code to use FlyCV during deployment.

You can turn on the FlyCV compile option when compiling the FastDeploy library as follows.
```bash
# When compiling C++ libraries
-DENABLE_VISION=ON

#  When compiling Python libraries
export ENABLE_FLYCV=ON
```

You can enable FlyCV by adding a new line of code to the deployment code as follows.
```bash
# C++ code
fastdeploy::vision::EnableFlyCV();
# Other..(e.g. With Huawei Ascend)
fastdeploy::RuntimeOption option;
option.UseAscend();
...


# Python code
fastdeploy.vision.enable_flycv()
# Other..(e.g. With Huawei Ascend)
runtime_option = build_option()
option.use_ascend()
...
```

## Some Platforms FlyCV End-to-End Inference Performance

KunPeng 920 CPU + Atlas 300I Pro.
| Model | OpenCV E2E Performance(ms) | FlyCV E2E Performance(ms) |  
| :-----------| :--------   | :--------   |
|   ResNet50     | 2.78  | 1.63  |  
|   PP-LCNetV2   |  2.50 |  1.39   |  
|   YOLOv7       |  27.00 | 21.36    |  
|   PP_HumanSegV2_Lite   | 2.76 |  2.10   |  


Rockchip RV1126.

| Model | OpenCV E2E Performance(ms) | FlyCV E2E Performance(ms) |  
| :-----------| :--------   | :--------   |
|   ResNet50     | 9.23  | 6.01  |  
|   mobilenetv1_ssld_量化模型   |  9.23 |  6.01   |  
|   yolov5s_量化模型       |  28.33 | 14.25    |  
|   PP_LiteSeg_量化模型  | 132.25 |  60.31   |  
