# 编译SCRFD示例

当前支持模型版本为：[SCRFD CID:17cdeab](https://github.com/deepinsight/insightface/tree/17cdeab12a35efcebc2660453a8cbeae96e20950)

本文档说明如何进行[SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)的快速部署推理。本目录结构如下

## 获取ONNX文件

- 手动获取

  访问[SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)官方github库，按照指引下载安装，下载`scrfd.pt` 模型，利用 `tools/scrfd2onnx.py` 得到`onnx`格式文件。


  ```
  #下载scrfd模型文件
  e.g. download from  https://onedrive.live.com/?authkey=%21ABbFJx2JMhNjhNA&id=4A83B6B633B029CC%215542&cid=4A83B6B633B029CC

  # 安装官方库配置环境，此版本导出环境为：
  - 手动配置环境
    torch==1.8.0
    mmcv==1.3.5
    mmdet==2.7.0

  - 通过docker配置
    docker pull qyjdefdocker/onnx-scrfd-converter:v0.3

  # 导出onnx格式文件
  - 手动生成
    python tools/scrfd2onnx.py configs/scrfd/scrfd_500m.py weights/scrfd_500m.pth --shape 640 --input-img face-xxx.jpg

  - docker
    docker的onnx目录中已有生成好的onnx文件


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
cp PATH/TO/yolov7.onnx PATH/TO/model_zoo/vision/yolov7/cpp/build/

# 下载图片
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# 执行
./yolov7_demo
```

执行完后可视化的结果保存在本地`vis_result.jpg`，同时会将检测框输出在终端，如下所示
```
FaceDetectionResult: [xmin, ymin, xmax, ymax, score]
437.670410,194.262772, 478.729828, 244.633911, 0.912465
418.303650,118.277687, 455.877838, 169.209564, 0.911748
269.449493,280.810608, 319.466614, 342.681213, 0.908530
775.553955,237.509979, 814.626526, 286.252350, 0.901296
565.155945,303.849670, 608.786255, 356.025726, 0.898307
411.813477,296.117584, 454.560394, 353.151367, 0.889968
688.620239,153.063812, 728.825195, 204.860321, 0.888146
686.523071,304.881104, 732.901245, 364.715088, 0.885789
194.658829,236.657883, 234.194748, 289.099701, 0.881143
137.273422,286.025787, 183.479523, 344.614441, 0.877399
289.256775,148.388992, 326.087769, 197.035645, 0.875090
182.943939,154.105682, 221.422440, 204.460495, 0.871119
330.301849,207.786499, 367.546692, 260.813232, 0.869559
659.884216,254.861847, 701.580017, 307.984711, 0.869249
550.305359,232.336868, 591.702026, 281.101532, 0.866158
567.473511,127.402367, 604.959839, 175.831696, 0.858938
```
