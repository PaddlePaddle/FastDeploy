# 编译ScaledYOLOv4示例

当前支持模型版本为：[ScaledYOLOv4 branch yolov4-large](https://github.com/WongKinYiu/ScaledYOLOv4)

本文档说明如何进行[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)的快速部署推理。本目录结构如下

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── README.md
│   └── scaledyolov4.cc
├── README.md
└── scaled_yolov4.py
```

## 获取ONNX文件

- 手动获取

  访问[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)官方github库，按照指引下载安装，下载`scaledyolov4.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。如果您导出的`onnx`模型出现问题，可以参考[ScaledYOLOv4#401](https://github.com/WongKinYiu/ScaledYOLOv4/issues/401)的解决办法

  ```
  #下载ScaledYOLOv4模型文件
  Download from the goole drive https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing

  # 导出onnx格式文件
  python models/export.py  --weights PATH/TO/scaledyolov4-xx.pt --img-size 640

  # 移动onnx文件到demo目录
  cp PATH/TO/scaledyolov4.onnx PATH/TO/model_zoo/vision/scaledyolov4/
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
python scaled_yolov4.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
665.666321,390.477173, 810.000000, 879.829346, 0.940627, 0
48.266064,396.217163, 247.338425, 901.974915, 0.922277, 0
221.351868,408.446259, 345.524017, 857.927917, 0.910516, 0
14.989746,228.662842, 801.292236, 735.677490, 0.820487, 5
0.000000,548.260864, 75.825439, 873.932495, 0.718777, 0
134.789062,473.950195, 148.526367, 506.777344, 0.513963, 27
```

## 其它文档

- [C++部署](./cpp/README.md)
- [ScaledYOLOv4 API文档](./api.md)
