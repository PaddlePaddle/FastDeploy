## 更新记录

2020.11.4
1. 支持Paddle动态图模型导出为ONNX。
2. 重构代码结构以更好地支持不同Paddle版本，以及动态图和静态图的转换。
3. 提升ONNX Opset的覆盖率，未来仍将稳定支持9，10，11三个版本。

2020.9.21
1. 支持ONNX Opset 9, 10, 11三个版本的导出。
2. 新增支持转换的OP: swish, floor, uniform_random, abs, instance_norm, clip, tanh, log, norm和pad2d。

2019.09.25
1. 新增支持SE_ResNet50_vd、SqueezeNet1_0、SE_ResNext50_32x4d、Xception41、VGG16、InceptionV4、YoloV3模型转换。
2. 解决0.1版本无法适配新版ONNX版本问题。

2019.08.20
1. 解决preview版本无法适配最新的PaddlePaddle和ONNX版本问题。
2. 功能上支持主流的图像分类模型和部分图像检测模型。
3. 统一对外的使用接口，用户可利用PIP安装功能包进行使用。
