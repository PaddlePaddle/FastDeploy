# 编译YOLOR示例

当前支持模型版本为：[YOLOR weights](https://github.com/WongKinYiu/yolor/releases/tag/weights)

## 获取ONNX文件

- 手动获取

  访问[YOLOR](https://github.com/WongKinYiu/yolor)官方github库，按照指引下载安装，下载`yolor.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。

  ```
  #下载yolor模型文件
  wget https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6-paper-570.pt

  # 导出onnx格式文件
  python models/export.py  --weights PATH/TO/yolor-xx-xx-xx.pt --img-size 640

  # 如果您在导出的`onnx`模型出现精度不达标或者是数据维度的问题，可以参考[@DefTruth](https://github.com/DefTruth/lite.ai.toolkit/blob/main/docs/ort/ort_yolor.zh.md)的解决办法

  # 移动onnx文件到demo目录
  cp PATH/TO/yolor.onnx PATH/TO/model_zoo/vision/yolor/
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
cp PATH/TO/yolor.onnx PATH/TO/model_zoo/vision/yolor/cpp/build/

# 下载图片
wget https://raw.githubusercontent.com/WongKinYiu/yolor/paper/inference/images/horses.jpg

# 执行
./yolor_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
DetectionResult: [xmin, ymin, xmax, ymax, score, label_id]
0.000000,185.201431, 315.673126, 410.071594, 0.959289, 17
433.802826,211.603455, 595.489319, 346.425537, 0.952615, 17
230.446854,195.618805, 418.365479, 362.712128, 0.884253, 17
336.545624,208.555618, 457.704315, 323.543152, 0.788450, 17
0.896423,183.936996, 154.788727, 304.916412, 0.672804, 17
```
