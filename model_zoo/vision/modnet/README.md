# MODNet 部署示例

## 0. 简介
当前支持模型版本为：[MODNet CommitID:28165a4](https://github.com/ZHKKKe/MODNet/commit/28165a4)

本文档说明如何进行[MODNet](https://github.com/ZHKKKe/MODNet) 的快速部署推理。本目录结构如下

```
.
├── cpp                     # C++ 代码目录
│   ├── CMakeLists.txt      # C++ 代码编译CMakeLists文件
│   ├── README.md           # C++ 代码编译部署文档
│   └── modnet.cc           # C++ 示例代码
├── api.md                  # API 说明文档
├── README.md               # MODNet 部署文档
└── modnet.py               # Python示例代码
```

## 1. 获取ONNX文件

访问[MODNet](https://github.com/ZHKKKe/MODNet)官方github库，按照指引下载安装，下载模型文件，利用 `onnx/export_onnx.py` 得到`onnx`格式文件。

* 导出onnx格式文件
  ```bash
  python -m onnx.export_onnx \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=pretrained/modnet_photographic_portrait_matting.onnx
  ```
* 移动onnx文件到model_zoo/modnet的目录
  ```bash
  cp PATH/TO/modnet_photographic_portrait_matting.onnx PATH/TO/model_zoo/vision/modnet/
  ```


## 2. 准备测试图片
准备1张仅包含人像的测试图片，命名为matting_1.jpg，并拷贝到可执行文件所在的目录，比如
```bash
matting_1.jpg
```

## 3. 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`
```bash
# 安装fastdeploy-python工具
pip install fastdeploy-python

# 安装vision-cpu模块
fastdeploy install vision-cpu
```

## 4. Python部署

执行如下代码即会自动下载MODNet模型和测试图片
```bash
python modnet.py
```

执行完成后会输出检测结果如下, 可视化结果保存在`vis_result.jpg`中
```
MattingResult[Foreground(false), Alpha(Numel(65536), Shape(256,256), Min(0.000000), Max(1.000000), Mean(0.464415))]
```

## 5. 其它文档

- [C++部署](./cpp/README.md)
- [MODNet API文档](./api.md)
