[English](README.md) | 简体中文
# PP-YOLOE  量化模型 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 PP-YOLOE 和 PicoDet 量化模型在 RV1126 上的部署推理加速。

## 部署准备
### FastDeploy 交叉编译环境准备
1. 软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 交叉编译环境准备](../../../../../../docs/cn/build_and_install/rv1126.md#交叉编译环境搭建)  

### 模型准备
1. 用户可以直接使用由 FastDeploy 提供的量化模型进行部署。
2. 用户可以先使用 PaddleDetection 自行导出 Float32 模型，注意导出 PP-YOLOE 模型时设置参数：use_shared_conv=False，更多细节请参考：[PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)，导出 PicoDet 请参考：[PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)
3. 用户可以使用 FastDeploy 提供的[一键模型自动化压缩工具](../../../../../../tools/common_tools/auto_compression/)，自行进行模型量化, 并使用产出的量化模型进行部署。（注意: 推理量化后的检测模型仍然需要FP32模型文件夹下的 infer_cfg.yml 文件，自行量化的模型文件夹内不包含此 yaml 文件，用户从 FP32 模型文件夹下复制此yaml文件到量化后的模型文件夹内即可。）
4. 模型需要异构计算，异构计算文件可以参考：[异构计算](./../../../../../../docs/cn/faq/heterogeneous_computing_on_timvx_npu.md)，由于 FastDeploy 已经提供了模型，可以先测试我们提供的异构文件，验证精度是否符合要求。

更多量化相关相关信息可查阅[模型量化](../../quantize/README.md)

## 在 RV1126 上部署量化后的 PP-YOLOE 和 PicoDet 检测模型
请按照以下步骤完成在 RV1126 上部署 PP-YOLOE 和 PicoDet 量化模型：
1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](../../../../../../docs/cn/build_and_install/rv1126.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
mkdir models && mkdir images
# 下载 FastDeploy 准备的 PP-YOLOE 模型
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_noshare_qat.tar.gz
tar -xvf ppyoloe_noshare_qat.tar.gz
cp -r ppyoloe_noshare_qat models
# 下载 FastDeploy 准备的 PicoDet 模型
wget https://bj.bcebos.com/fastdeploy/models/picodet_withNMS_quant_qat.tar.gz
tar -xvf picodet_withNMS_quant_qat.tar.gz
cp -r picodet_withNMS_quant_qat models
# 下载 FastDeploy 准备的测试图片
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. 编译部署示例，可使入如下命令：
```bash
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 PP-YOLOE 和 PicoDet 检测模型到 Rockchip RV1126，可使用如下命令：
```bash
# 进入 install 目录
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh ppyoloe_infer_demo ppyoloe_noshare_qat 000000014439.jpg $DEVICE_ID
bash run_with_adb.sh picodet_infer_demo picodet_withNMS_quant_qat 000000014439.jpg $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/203708564-43c49485-9b48-4eb2-8fe7-0fa517979fff.png">

需要特别注意的是，在 RV1126 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](../../../../../../docs/cn/quantize.md)
