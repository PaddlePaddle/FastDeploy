# PaddleDetection/PPYOLOE部署示例

- 当前支持PaddleDetection版本为[release/2.4](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)

本文档说明如何进行[PPYOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)的快速部署推理。本目录结构如下
```
.
├── cpp                 # C++ 代码目录
│   ├── CMakeLists.txt  # C++ 代码编译CMakeLists文件
│   ├── README.md       # C++ 代码编译部署文档
│   └── ppyoloe.cc      # C++ 示例代码
├── README.md           # PPYOLOE 部署文档
└── ppyoloe.py          # Python示例代码
```

## 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`
```
# 安装fastdeploy-python工具
pip install fastdeploy-python
```

## Python部署

执行如下代码即会自动下载PPYOLOE模型和测试图片
```
python ppyoloe.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
162.380249,132.057449, 463.178345, 413.167114, 0.962918, 33
414.914642,141.148666, 91.275269, 308.688293, 0.951003, 0
163.449234,129.669067, 35.253891, 135.111786, 0.900734, 0
267.232239,142.290436, 31.578918, 126.329773, 0.848709, 0
581.790833,179.027115, 30.893127, 135.484940, 0.837986, 0
104.407021,72.602615, 22.900627, 75.469055, 0.796468, 0
348.795380,70.122147, 18.806061, 85.829330, 0.785557, 0
364.118683,92.457428, 17.437622, 89.212891, 0.774282, 0
75.180283,192.470490, 41.898407, 55.552414, 0.712569, 56
328.133759,61.894299, 19.100616, 65.633575, 0.710519, 0
504.797760,181.732574, 107.740814, 248.115082, 0.708902, 0
379.063080,64.762360, 15.956146, 68.312546, 0.680725, 0
25.858747,186.564178, 34.958130, 56.007080, 0.580415, 0
```

## 其它文档

- [C++部署](./cpp/README.md)
- [PPYOLOE API文档](./api.md)
