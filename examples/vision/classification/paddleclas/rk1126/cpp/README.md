# PaddleClas RK1126开发板 C++ 部署示例
本目录下提供的 `infer.cc`，可以帮助用户快速完成 PaddleClas 量化模型在 RK1126 上的部署推理加速。

## 部署准备
### FastDeploy 交叉编译环境准备
- 1. 软硬件环境满足要求，以及交叉编译环境的准备，请参考：[FastDeploy 交叉编译环境准备](../../../../../../docs/cn/build_and_install/rk1126.md#交叉编译环境搭建)  

### 量化模型准备
- 1. 用户可以直接使用由 FastDeploy 提供的量化模型进行部署。
- 2. 用户可以使用 FastDeploy 提供的[一键模型自动化压缩工具](../../../../../../tools/auto_compression/)，自行进行模型量化, 并使用产出的量化模型进行部署。(注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的inference_cls.yaml文件, 自行量化的模型文件夹内不包含此 yaml 文件, 用户从 FP32 模型文件夹下复制此 yaml 文件到量化后的模型文件夹内即可.)
- 更多量化相关相关信息可查阅[模型量化](../../quantize/README.md)

## 在 RK1126 上部署量化后的 ResNet50_Vd 分类模型
请按照以下步骤完成在 RK1126 上部署 ResNet50_Vd 量化模型：
1. 交叉编译编译 FastDeploy 库，具体请参考：[交叉编译 FastDeploy](../../../../../../docs/cn/build_and_install/rk1126.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-tmivx/ FastDeploy/examples/vision/classification/paddleclas/rk1126/cpp/
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
mkdir models && mkdir images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
cp -r ResNet50_vd_infer models
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
cp -r ILSVRC2012_val_00000010.jpeg images
```

4. 编译部署示例，可使入如下命令：
```bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../fastdeploy-tmivx/timvx.cmake -DFASTDEPLOY_INSTALL_DIR=fastdeploy-tmivx ..
make -j8
make install
# 成功编译之后，会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 基于 adb 工具部署 ResNet50_vd 分类模型到 Rockchip RV1126，可使用如下命令：
```bash
# 进入 install 目录
cd FastDeploy/examples/vision/classification/paddleclas/rk1126/cpp/build/install/
# 如下命令表示：bash run_with_adb.sh 需要运行的demo 模型路径 图片路径 设备的DEVICE_ID
bash run_with_adb.sh infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg $DEVICE_ID
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/200767389-26519e50-9e4f-4fe1-8d52-260718f73476.png">

需要特别注意的是，在 RK1126 上部署的模型需要是量化后的模型，模型的量化请参考：[模型量化](../../../../../../docs/cn/quantize.md)
