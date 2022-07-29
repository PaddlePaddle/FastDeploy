# PPOCR-DBDetector部署示例

当前支持模型版本为：[ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/models_list.md#1.1)

本文档说明如何进行[ch_PP-OCRv3_det](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/models_list.md#1.1)的快速部署推理,目前仅支持检测模型。本目录结构如下
```
.
├── cpp                 # C++ 代码目录
│   ├── CMakeLists.txt  # C++ 代码编译CMakeLists文件
│   ├── README.md       # C++ 代码编译部署文档
│   └── dbdetector.cc       # C++ 示例代码
├── README.md           # dbdetector 部署文档
└── dbdetector.py           # Python示例代码
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

执行如下代码即会自动下载PPOCR-dbdetector模型和测试图片
```
python dbdetector.py
```

执行完成后会将可视化结果保存在本地`vis_result.jpg`，同时输出检测结果如下
```
det boxes: [[71,549],[431,539],[432,575],[72,585]], det boxes: [[17,504],[518,482],[521,533],[20,554]], det boxes: [[184,454],[401,445],[402,482],[185,491]], det boxes: [[36,409],[487,385],[490,432],[39,456]]
```

## 其它文档

- [C++部署](./cpp/README.md)
- [DBDetector API文档](./api.md)
