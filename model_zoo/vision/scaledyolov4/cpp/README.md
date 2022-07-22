# 编译ScaledYOLOv4示例

当前支持模型版本为：[ScaledYOLOv4 branch yolov4-large](https://github.com/WongKinYiu/ScaledYOLOv4)
## 获取ONNX文件

- 手动获取

  访问[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)官方github库，按照指引下载安装，下载`scaledyolov4.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。如果您导出的`onnx`模型出现问题，可以参考[ScaledYOLOv4#401](https://github.com/WongKinYiu/ScaledYOLOv4/issues/401)的解决办法

  ```
  #下载ScaledYOLOv4模型文件
  Download from the goole drive https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing

  # 导出onnx格式文件
  python models/export.py  --weights PATH/TO/scaledyolov4-xx-xx-xx.pt --img-size 640

  # 移动onnx文件到demo目录
  cp PATH/TO/scaledyolov4.onnx PATH/TO/model_zoo/vision/scaledyolov4/
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
cp PATH/TO/scaledyolov4.onnx PATH/TO/model_zoo/vision/scaledyolov4/cpp/build/

# 下载图片
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg

# 执行
./scaledyolov4_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
665.666321,390.477173, 810.000000, 879.829346, 0.940627, 0
48.266064,396.217163, 247.338425, 901.974915, 0.922277, 0
221.351868,408.446259, 345.524017, 857.927917, 0.910516, 0
14.989746,228.662842, 801.292236, 735.677490, 0.820487, 5
0.000000,548.260864, 75.825439, 873.932495, 0.718777, 0
134.789062,473.950195, 148.526367, 506.777344, 0.513963, 27
```
