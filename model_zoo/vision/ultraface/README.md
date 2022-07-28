# UltraFace部署示例

当前支持模型版本为：[UltraFace CommitID:dffdddd](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd)

本文档说明如何进行[UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/)的快速部署推理。本目录结构如下

```
.
├── cpp                     # C++ 代码目录
│   ├── CMakeLists.txt      # C++ 代码编译CMakeLists文件
│   ├── README.md           # C++ 代码编译部署文档
│   └── ultraface.cc        # C++ 示例代码
├── api.md                  # API 说明文档
├── README.md               # UltraFace 部署文档
└── ultraface.py            # Python示例代码
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

执行如下代码即会自动下载UltraFace模型和测试图片
```bash
python ultraface.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
FaceDetectionResult: [xmin, ymin, xmax, ymax, score]
742.528931,261.309937, 837.749146, 365.145599, 0.999833
408.159332,253.410889, 484.747284, 353.378052, 0.999832
549.409424,225.051819, 636.311890, 337.824707, 0.999782
185.562805,233.364044, 252.001801, 323.948669, 0.999709
304.065918,180.468140, 377.097961, 278.932861, 0.999645
```

## 其它文档

- [C++部署](./cpp/README.md)
- [UltraFace API文档](./api.md)
