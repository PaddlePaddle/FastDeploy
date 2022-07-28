# RetinaFace部署示例

当前支持模型版本为：[RetinaFace CommitID:b984b4b](https://github.com/biubug6/Pytorch_Retinaface/commit/b984b4b)

本文档说明如何进行[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)的快速部署推理。本目录结构如下

```
.
├── cpp                     # C++ 代码目录
│   ├── CMakeLists.txt      # C++ 代码编译CMakeLists文件
│   ├── README.md           # C++ 代码编译部署文档
│   └── retinaface.cc       # C++ 示例代码
├── api.md                  # API 说明文档
├── README.md               # RetinaFace 部署文档
└── retinaface.py           # Python示例代码
```

## 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`
```bash
# 安装fastdeploy-python工具
pip install fastdeploy-python

# 安装vision-cpu模块
fastdeploy install vision-cpu
```

## Python部署

执行如下代码即会自动下载RetinaFace模型和测试图片
```bash
python retinaface.py
```

## 手动获取ONNX模型文件
自动下载的模型文件是我们事先转换好的，如果您需要从RetinaFace官方repo导出ONNX，请参考以下步骤。  

* 下载官方仓库并
```bash
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```
* 下载预训练权重并放在weights文件夹
```text
./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
* 运行convert_to_onnx.py导出ONNX模型文件
```bash
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 640 --cpu
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/Resnet50_Final.pth --network resnet50 --long_side 640 --cpu
```
注意：需要先对convert_to_onnx.py脚本中的--long_side参数增加类型约束，type=int.
* 使用onnxsim对模型进行简化
```bash
onnxsim FaceDetector.onnx Pytorch_RetinaFace_mobile0.25-640-640.onnx  # mobilenet
onnxsim FaceDetector.onnx Pytorch_RetinaFace_resnet50-640-640.onnx  # resnet50
```


执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
FaceDetectionResult: [xmin, ymin, xmax, ymax, score, (x, y) x 5]
403.339783,254.192413, 490.002747, 351.931213, 0.999427, (425.657257,293.820740), (467.249451,293.667267), (446.830078,315.016388), (428.903381,326.129425), (465.764648,325.837341)
296.834564,181.992035, 384.516876, 277.461243, 0.999194, (313.605164,224.800110), (352.888977,219.088043), (333.530182,239.872787), (325.395203,255.463852), (358.417175,250.529892)
742.206238,263.547424, 840.871765, 366.171387, 0.999068, (762.715759,308.939880), (809.019653,304.544830), (786.174194,329.286163), (771.952271,341.376038), (812.717529,337.528839)
545.351685,228.015930, 635.423584, 335.458649, 0.998681, (559.295654,269.971619), (598.439758,273.823608), (567.496643,292.894348), (558.160034,306.637238), (592.175781,309.493591)
180.078125,241.787888, 257.213135, 320.321777, 0.998342, (203.702591,272.032715), (237.497726,271.356445), (222.380402,288.225708), (208.015259,301.360352), (233.943451,300.801636)
```

## 其它文档

- [C++部署](./cpp/README.md)
- [RetinaFace API文档](./api.md)
