[English](README.md) |  简体中文
# YOLOv5 量化模型 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 YOLOv5 量化模型在 A311D 上的部署推理加速。

## 部署准备
### FastDeploy 交叉编译环境准备
1. 软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 交叉编译环境准备](../../../../../../docs/cn/build_and_install/a311d.md#交叉编译环境搭建)  

### 量化模型准备
可以直接使用由 FastDeploy 提供的量化模型进行部署，也可以按照如下步骤准备量化模型：
1. 按照 [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) 官方导出方式导出 ONNX 模型，或者直接使用如下命令下载
```bash
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx
```
2. 准备 300 张左右量化用的图片，也可以使用如下命令下载我们准备好的数据。
```bash
wget https://bj.bcebos.com/fastdeploy/models/COCO_val_320.tar.gz
tar -xf COCO_val_320.tar.gz
```
3. 使用 FastDeploy 提供的[一键模型自动化压缩工具](../../../../../../tools/common_tools/auto_compression/),自行进行模型量化, 并使用产出的量化模型进行部署。
```bash
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model_new/'
```
4. YOLOv5 模型需要异构计算，异构计算文件可以参考：[异构计算](./../../../../../../docs/cn/faq/heterogeneous_computing_on_timvx_npu.md)，由于 FastDeploy 已经提供了 YOLOv5 模型，可以先测试我们提供的异构文件，验证精度是否符合要求。
```bash
# 先下载我们提供的模型，解压后将其中的 subgraph.txt 文件拷贝到新量化的模型目录中
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
```

更多量化相关相关信息可查阅[模型量化](../../quantize/README.md)

## 在 A311D 上部署量化后的 YOLOv5 检测模型
请按照以下步骤完成在 A311D 上部署 YOLOv5 量化模型：
1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](../../../../../../docs/cn/build_and_install/a311d.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/yolov5/a311d/cpp
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
cp -r yolov5s_ptq_model models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. 编译部署示例，可使入如下命令：
```bash
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 YOLOv5 检测模型到晶晨 A311D
```bash
# 进入 install 目录
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo yolov5s_ptq_model 000000014439.jpg $DEVICE_ID
```

部署成功后，vis_result.jpg 保存的结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/203706969-dd58493c-6635-4ee7-9421-41c2e0c9524b.png">

需要特别注意的是，在 A311D 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](../../../../../../docs/cn/quantize.md)
