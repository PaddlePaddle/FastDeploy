# YOLOv6部署示例

当前支持模型版本为：[YOLOv6 v0.1.0](https://github.com/meituan/YOLOv6/releases/download/0.1.0)

本文档说明如何进行[YOLOv6](https://github.com/meituan/YOLOv6)的快速部署推理。本目录结构如下
```
.
├── cpp                 # C++ 代码目录
│   ├── CMakeLists.txt  # C++ 代码编译CMakeLists文件
│   ├── README.md       # C++ 代码编译部署文档
│   └── yolov6.cc       # C++ 示例代码
├── README.md           # YOLOv6 部署文档
└── yolov6.py           # Python示例代码
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

执行如下代码即会自动下载YOLOv6模型和测试图片
```
python yolov6.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
11.772949,229.269287, 792.933838, 748.294189, 0.954794, 5
667.140381,396.185455, 807.701721, 881.810120, 0.900997, 0
223.271011,405.105743, 345.740723, 859.328552, 0.898938, 0
50.135777,405.863129, 245.485519, 904.153809, 0.888936, 0
0.000000,549.002869, 77.864723, 869.455017, 0.614145, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOv6 API文档](./api.md)
