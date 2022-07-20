# 编译YOLOR示例

当前支持模型版本为：[YOLOR weights](https://github.com/WongKinYiu/yolor/releases/tag/weights)
(tips: 如果使用 `git clone` 的方式下载仓库代码，请将分支切换(checkout)到 `paper` 分支).

本文档说明如何进行[YOLOR](https://github.com/WongKinYiu/yolor)的快速部署推理。本目录结构如下

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── README.md
│   └── yolor.cc
├── README.md
└── yolor.py
```

## 获取ONNX文件

- 手动获取

  访问[YOLOR](https://github.com/WongKinYiu/yolor)官方github库，按照指引下载安装，下载`yolor.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。

  ```
  #下载yolor模型文件
  wget https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6-paper-570.pt

  # 导出onnx格式文件
  python models/export.py  --weights PATH/TO/yolor-xx-xx-xx.pt --img-size 640

  # 如果您在导出的`onnx`模型出现精度不达标或者是数据维度的问题，可以参考[@DefTruth](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_yolor.zh.md)的解决办法

  # 移动onnx文件到demo目录
  cp PATH/TO/yolor.onnx PATH/TO/model_zoo/vision/yolor/
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
python yolor.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
0.000000,185.201431, 315.673126, 410.071594, 0.959289, 17
433.802826,211.603455, 595.489319, 346.425537, 0.952615, 17
230.446854,195.618805, 418.365479, 362.712128, 0.884253, 17
336.545624,208.555618, 457.704315, 323.543152, 0.788450, 17
0.896423,183.936996, 154.788727, 304.916412, 0.672804, 17
```

## 其它文档

- [C++部署](./cpp/README.md)
- [YOLOR API文档](./api.md)
