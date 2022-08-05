# ArcFace部署示例

## 0. 简介
当前支持模型版本为：[ArcFace CommitID:babb9a5](https://github.com/deepinsight/insightface/commit/babb9a5)

本文档说明如何进行[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 的快速部署推理。本目录结构如下

```
.
├── cpp                     # C++ 代码目录
│   ├── CMakeLists.txt      # C++ 代码编译CMakeLists文件
│   ├── README.md           # C++ 代码编译部署文档
│   └── arcface.cc          # C++ 示例代码
├── api.md                  # API 说明文档
├── README.md               # ArcFace 部署文档
└── arcface.py              # Python示例代码
```

## 1. 特别说明  
fastdeploy支持 [insightface](https://github.com/deepinsight/insightface/tree/master/recognition) 的人脸识别模块recognition中大部分模型的部署，包括ArcFace、CosFace、Partial FC、VPL等，由于用法类似，这里仅用ArcFace来演示部署流程。所有支持的模型结构，可参考 [ArcFace API文档](./api.md).


## 2. 获取ONNX文件

访问[ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)官方github库，按照指引下载安装，下载pt模型文件，利用 `torch2onnx.py` 得到`onnx`格式文件。

* 下载ArcFace模型文件
  ```
  Link: https://pan.baidu.com/share/init?surl=CL-l4zWqsI1oDuEEYVhj-g code: e8pw  
  ```

* 导出onnx格式文件
  ```bash
  PYTHONPATH=. python ./torch2onnx.py partial_fc/pytorch/ms1mv3_arcface_r100_fp16/backbone.pth --output ms1mv3_arcface_r100.onnx --network r100 --simplify 1
  ```
* 移动onnx文件到model_zoo/arcface的目录
  ```bash
  cp PATH/TO/ms1mv3_arcface_r100.onnx PATH/TO/model_zoo/vision/arcface/
  ```


## 3. 准备测试图片
准备3张仅包含人脸的测试图片，命名为face_recognition_*.jpg，并拷贝到可执行文件所在的目录，比如
```bash
face_recognition_0.png  # 0,1 同一个人
face_recognition_1.png
face_recognition_2.png  # 0,2 不同的人
```

## 4. 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`
```bash
# 安装fastdeploy-python工具
pip install fastdeploy-python

# 安装vision-cpu模块
fastdeploy install vision-cpu
```

## 5. Python部署

执行如下代码即会自动下载ArcFace模型和测试图片
```bash
python arcface.py
```

执行完成后会输出检测结果如下
```
FaceRecognitionResult: [Dim(512), Min(-0.141219), Max(0.121645), Mean(-0.003172)]
FaceRecognitionResult: [Dim(512), Min(-0.117939), Max(0.141897), Mean(0.000407)]
FaceRecognitionResult: [Dim(512), Min(-0.124471), Max(0.112567), Mean(-0.001320)]
Cosine 01:  0.7211584683376316
Cosine 02:  -0.06262668682788906
```

## 6. 其它文档

- [C++部署](./cpp/README.md)
- [ArcFace API文档](./api.md)
