# 编译ArcFace示例

## 0. 简介
当前支持模型版本为：[MODNet CommitID:28165a4](https://github.com/ZHKKKe/MODNet/commit/28165a4)

## 1. 下载和解压预测库
```bash
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.3.0.tgz
tar xvf fastdeploy-linux-x64-0.3.0.tgz
```

## 2. 编译示例代码
```bash
mkdir build & cd build
cmake ..
make -j
```

## 3. 获取ONNX文件

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


## 4. 准备测试图片
准备1张仅包含人像的测试图片，命名为matting_1.jpg，并拷贝到可执行文件所在的目录，比如
```bash
matting_1.jpg
```

## 5. 执行
```bash
./modnet_demo
```

执行完成后会输出检测结果如下, 可视化结果保存在`vis_result.jpg`中
```
MattingResult[Foreground(false), Alpha(Numel(65536), Shape(256,256), Min(0.000000), Max(1.000000), Mean(0.464415))]
```
