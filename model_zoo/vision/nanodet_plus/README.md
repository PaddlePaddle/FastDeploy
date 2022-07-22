# NanoDetPlus部署示例

当前支持模型版本为：[NanoDetPlus v1.0.0-alpha-1](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1)

本文档说明如何进行[NanoDetPlus](https://github.com/RangiLyu/nanodet)的快速部署推理。本目录结构如下
```
.
├── cpp                  # C++ 代码目录
│   ├── CMakeLists.txt   # C++ 代码编译CMakeLists文件
│   ├── README.md        # C++ 代码编译部署文档
│   └── nanodet_plus.cc  # C++ 示例代码
├── README.md            # YOLOX 部署文档
└── nanodet_plus.py      # Python示例代码
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

执行如下代码即会自动下载NanoDetPlus模型和测试图片
```
python nanodet_plus.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
5.710144,220.634033, 807.854370, 724.089111, 0.825635, 5
45.646439,393.694061, 229.267044, 903.998413, 0.818263, 0
218.289322,402.268829, 342.083252, 861.766479, 0.709301, 0
698.587036,325.627197, 809.000000, 876.990967, 0.630235, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [NanoDetPlus API文档](./api.md)
