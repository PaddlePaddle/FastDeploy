# 编译YOLOv7示例

当前支持模型版本为：[YOLOv7 v0.1](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)

## 获取ONNX文件

- 手动获取

  访问[YOLOv7](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1)官方github库，按照指引下载安装，下载`yolov7.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。

  ```
  #下载yolov7模型文件
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

  # 导出onnx格式文件
  python models/export.py --grid --dynamic --weights PATH/TO/yolo7.pt

  ```


## 运行demo

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 移动onnx文件到demo目录
cp PATH/TO/yolo7.onnx PATH/TO/model_zoo/vision/yolov7/cpp/build/

# 下载图片
wget https://raw.githubusercontent.com/WongKinYiu/yolov7/main/inference/images/horses.jpg

# 执行
./yolov7_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
0.056616,191.221619, 314.871063, 409.948914, 0.955449, 17
432.547852,211.914841, 594.904297, 346.708618, 0.942706, 17
0.000000,185.456207, 153.967789, 286.157562, 0.860487, 17
224.049210,195.147003, 419.658234, 364.004852, 0.798262, 17
369.316986,209.055725, 456.373840, 321.627625, 0.687066, 17
```
