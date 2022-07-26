# 编译NanoDetPlus示例

当前支持模型版本为：[NanoDetPlus v1.0.0-alpha-1](https://github.com/RangiLyu/nanodet/releases/tag/v1.0.0-alpha-1)

```
# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j

# 下载模型和图片
wget https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx
wget https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg

# 执行
./nanodet_plus_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
5.710144,220.634033, 807.854370, 724.089111, 0.825635, 5
45.646439,393.694061, 229.267044, 903.998413, 0.818263, 0
218.289322,402.268829, 342.083252, 861.766479, 0.709301, 0
698.587036,325.627197, 809.000000, 876.990967, 0.630235, 0
```
