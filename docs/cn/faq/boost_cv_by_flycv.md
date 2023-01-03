[English](../../en/faq/boost_cv_by_flycv.md) | 中文


# 使用FlyCV加速端到端推理性能

[FlyCV](https://github.com/PaddlePaddle/FlyCV) 是一款高性能计算机图像处理库, 针对ARM架构做了很多优化, 相比其他图像处理库性能更为出色.
FastDeploy现在已经集成FlyCV, 用户可以在支持的硬件平台上使用FlyCV, 实现模型端到端推理性能的加速.

## 已支持的系统与硬件架构

| 系统 | 硬件架构 |
| :-----------| :--------   |
|   Android     |  armeabi-v7a, arm64-v8a |  
|   Linux       |  aarch64, armhf, x86_64|  


## 使用方式
使用FlyCV,首先需要在编译时开启FlyCV编译选项,之后在部署时新增一行代码即可开启.
本文以Linux系统为例,说明如何开启FlyCV编译选项, 之后在部署时, 新增一行代码使用FlyCV.

用户可以按照如下方式,在编译预测库时,开启FlyCV编译选项.
```bash
# 编译C++预测库时, 开启FlyCV编译选项.
-DENABLE_VISION=ON \

# 在编译Python预测库时, 开启FlyCV编译选项
export ENABLE_FLYCV=ON
```

用户可以按照如下方式,在部署代码中新增一行代码启用FlyCV.
```bash
# C++部署代码.
# 新增一行代码启用FlyCV
fastdeploy::vision::EnableFlyCV();
# 其他部署代码...(以昇腾部署为例)
fastdeploy::RuntimeOption option;
option.UseAscend();
...


# Python部署代码
# 新增一行代码启用FlyCV
fastdeploy.vision.enable_flycv()
# 其他部署代码...(以昇腾部署为例)
runtime_option = build_option()
option.use_ascend()
...
```

## 部分平台FlyCV 端到端性能数据

鲲鹏920 CPU + Atlas 300I Pro 推理卡.
| 模型 | OpenCV 端到端性能(ms) | FlyCV 端到端性能(ms) |  
| :-----------| :--------   | :--------   |
|   ResNet50     | 2.78  | 1.63  |  
|   PP-LCNetV2   |  2.50 |  1.39   |  
|   YOLOv7       |  27.00 | 21.36    |  
|   PP_HumanSegV2_Lite   | 2.76 |  2.10   |  


瑞芯微RV1126.

| 模型 | OpenCV 端到端性能(ms) | FlyCV 端到端性能(ms) |  
| :-----------| :--------   | :--------   |
|   ResNet50     | 9.23  | 6.01  |  
|   mobilenetv1_ssld_量化模型   |  9.23 |  6.01   |  
|   yolov5s_量化模型       |  28.33 | 14.25    |  
|   PP_LiteSeg_量化模型  | 132.25 |  60.31   |  
