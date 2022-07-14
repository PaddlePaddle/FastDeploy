# 编译YOLOv5示例


```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 下载模型和图片
wget "TODO"
wget https://github.com/WongKinYiu/yolov7/blob/main/inference/images/horses.jpg

# 执行
./yolov7_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
223.395142,403.948669, 345.337189, 867.339050, 0.856906, 0
668.301758,400.781342, 808.441772, 882.534973, 0.829716, 0
50.210720,398.571411, 243.123367, 905.016602, 0.805375, 0
23.768242,214.979370, 802.627686, 778.840881, 0.756311, 5
0.737200,552.281006, 78.617218, 890.945007, 0.363471, 0
```
