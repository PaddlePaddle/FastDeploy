# 编译YOLOv7示例

当前支持模型版本为：[YOLOv7 v0.1](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)

本文档说明如何进行[YOLOv7](https://github.com/WongKinYiu/yolov7)的快速部署推理。本目录结构如下

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── README.md
│   └── yolov7.cc
├── README.md
└── yolov7.py
```

## 获取ONNX文件

- 手动获取

  访问[YOLOv7](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)官方github库，按照指引下载安装，下载`yolov7.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。



  ```
  #下载yolov7模型文件
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

  # 导出onnx格式文件
  python models/export.py --grid --dynamic --weights PATH/TO/yolo7.pt

  # 移动onnx文件到demo目录
  cp PATH/TO/yolo7.onnx PATH/TO/model_zoo/vision/yolov7/
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

执行如下代码即会自动下载测试图片
```
python yolov7.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
0.056616,191.221619, 314.871063, 409.948914, 0.955449, 17
432.547852,211.914841, 594.904297, 346.708618, 0.942706, 17
0.000000,185.456207, 153.967789, 286.157562, 0.860487, 17
224.049210,195.147003, 419.658234, 364.004852, 0.798262, 17
369.316986,209.055725, 456.373840, 321.627625, 0.687066, 17
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOv7 API文档](./api.md)
