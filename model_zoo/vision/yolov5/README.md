# YOLOv5部署示例

当前支持模型版本为：[YOLOv5 v6.0](https://github.com/ultralytics/yolov5/releases/download/v6.0)

本文档说明如何进行[YOLOv5](https://github.com/ultralytics/yolov5)的快速部署推理。本目录结构如下
```
.
├── cpp                 # C++ 代码目录
│   ├── CMakeLists.txt  # C++ 代码编译CMakeLists文件
│   ├── README.md       # C++ 代码编译部署文档
│   └── yolov5.cc       # C++ 示例代码
├── README.md           # YOLOv5 部署文档
└── yolov5.py           # Python示例代码
```

## 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`
```
# 安装fastdeploy-python工具
pip install fastdeploy-python

# 安装vision-cpu模块
fastdeploy install vision-cpu
```

## Python部署

执行如下代码即会自动下载YOLOv5模型和测试图片
```
python yolov5.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
223.395142,403.948669, 345.337189, 867.339050, 0.856906, 0
668.301758,400.781342, 808.441772, 882.534973, 0.829716, 0
50.210720,398.571411, 243.123367, 905.016602, 0.805375, 0
23.768242,214.979370, 802.627686, 778.840881, 0.756311, 5
0.737200,552.281006, 78.617218, 890.945007, 0.363471, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOv5 API文档](./api.md)
