# 编译YOLOv6示例

当前支持模型版本为：[YOLOv6 v0.1.0](https://github.com/meituan/YOLOv6/releases/download/0.1.0)

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 下载模型和图片
wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg

# 执行
./yolov6_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
11.772949,229.269287, 792.933838, 748.294189, 0.954794, 5
667.140381,396.185455, 807.701721, 881.810120, 0.900997, 0
223.271011,405.105743, 345.740723, 859.328552, 0.898938, 0
50.135777,405.863129, 245.485519, 904.153809, 0.888936, 0
0.000000,549.002869, 77.864723, 869.455017, 0.614145, 0
```
