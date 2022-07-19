# YOLOX部署示例

当前支持模型版本为：[YOLOX v0.1.1](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0)

本文档说明如何进行[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)的快速部署推理。本目录结构如下
```
.
├── cpp                 # C++ 代码目录
│   ├── CMakeLists.txt  # C++ 代码编译CMakeLists文件
│   ├── README.md       # C++ 代码编译部署文档
│   └── yolox.cc        # C++ 示例代码
├── README.md           # YOLOX 部署文档
└── yolox.py            # Python示例代码
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

执行如下代码即会自动下载YOLOX模型和测试图片
```
python yolox.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
17.151855,225.294434, 805.329712, 735.578613, 0.940478, 5
671.162109,387.403961, 809.000000, 879.525513, 0.909566, 0
54.373432,400.188110, 204.652756, 893.662537, 0.894507, 0
221.339310,406.614960, 347.045593, 857.299927, 0.887144, 0
0.083759,554.987305, 61.894527, 881.098816, 0.450202, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOX API文档](./api.md)
