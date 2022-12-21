# YOLOv5 量化模型 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 YOLOv5 量化模型在 RV1126 上的部署推理加速。

## 部署准备
### FastDeploy 交叉编译环境准备
- 1. 软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 交叉编译环境准备](../../../../../../docs/cn/build_and_install/rv1126.md#交叉编译环境搭建)  

### 量化模型准备
- 1. 用户可以直接使用由 FastDeploy 提供的量化模型进行部署。
- 2. 用户可以使用 FastDeploy 提供的[一键模型自动化压缩工具](../../../../../../tools/common_tools/auto_compression/),自行进行模型量化, 并使用产出的量化模型进行部署。
- 更多量化相关相关信息可查阅[模型量化](../../quantize/README.md)

## 在 RV1126 上部署量化后的 YOLOv5 检测模型
请按照以下步骤完成在 RV1126 上部署 YOLOv5 量化模型：
1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](../../../../../../docs/cn/build_and_install/rv1126.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
cp -r yolov5s_ptq_model models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. 编译部署示例，可使入如下命令：
```bash
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 YOLOv5 检测模型到 Rockchip RV1126，可使用如下命令：
```bash
# 进入 install 目录
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo yolov5s_ptq_model 000000014439.jpg $DEVICE_ID
```

部署成功后，vis_result.jpg 保存的结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/203706969-dd58493c-6635-4ee7-9421-41c2e0c9524b.png">

需要特别注意的是，在 RV1126 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](../../../../../../docs/cn/quantize.md)
