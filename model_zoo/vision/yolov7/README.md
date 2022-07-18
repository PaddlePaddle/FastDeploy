# 编译YOLOv7示例

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

## 生成ONNX文件

- 手动获取

  访问[YOLOv7](https://github.com/WongKinYiu/yolov7)官方github库，按照指引下载安装，下载`yolov7.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。

  

  ```
  #下载yolov7模型文件
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  
  # 导出onnx格式文件
  python models/export.py --grid --dynamic --weights PATH/TO/yolo7.pt
  ```

  

- 从PaddlePaddle获取



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
223.395142,403.948669, 345.337189, 867.339050, 0.856906, 0
668.301758,400.781342, 808.441772, 882.534973, 0.829716, 0
50.210720,398.571411, 243.123367, 905.016602, 0.805375, 0
23.768242,214.979370, 802.627686, 778.840881, 0.756311, 5
0.737200,552.281006, 78.617218, 890.945007, 0.363471, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOv7 API文档](./api.md)






