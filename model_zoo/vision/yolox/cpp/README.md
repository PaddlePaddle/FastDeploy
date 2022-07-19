# 编译YOLOX示例

当前支持模型版本为：[YOLOX v0.1.1](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0)

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 下载模型和图片
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg

# 执行
./yolox_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
17.151855,225.294434, 805.329712, 735.578613, 0.940478, 5
671.162109,387.403961, 809.000000, 879.525513, 0.909566, 0
54.373432,400.188110, 204.652756, 893.662537, 0.894507, 0
221.339310,406.614960, 347.045593, 857.299927, 0.887144, 0
0.083759,554.987305, 61.894527, 881.098816, 0.450202, 0
```
