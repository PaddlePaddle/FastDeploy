# 编译UltraFace示例

当前支持模型版本为：[UltraFace CommitID:dffdddd](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd)

## 下载和解压预测库
```bash
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz
```

## 编译示例代码
```bash
mkdir build & cd build
cmake ..
make -j
```

## 下载模型和图片
wget https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx  
wget https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/imgs/3.jpg


## 执行
```bash
./ultraface_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
FaceDetectionResult: [xmin, ymin, xmax, ymax, score]
742.528931,261.309937, 837.749146, 365.145599, 0.999833
408.159332,253.410889, 484.747284, 353.378052, 0.999832
549.409424,225.051819, 636.311890, 337.824707, 0.999782
185.562805,233.364044, 252.001801, 323.948669, 0.999709
304.065918,180.468140, 377.097961, 278.932861, 0.999645
```
