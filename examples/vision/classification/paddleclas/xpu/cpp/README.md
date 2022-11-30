# PaddleClas 模型在昆仑芯 XPU 上的 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 PaddleClas 模型在昆仑芯 XPU 上的部署推理加速。

## 在昆仑芯 XPU 上部署 ResNet50_Vd 分类模型
请按照以下步骤完成在昆仑芯 XPU 上部署 ResNet50_Vd 检测模型：
1. 适用于昆仑芯 XPU 的 FastDeploy 库编译，具体请参考：[编译 FastDeploy](../../../../../../docs/cn/build_and_install/xpu.md)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-xpu/ FastDeploy/examples/vision/classification/paddleclas/xpu/cpp/
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
```

4. 编译部署示例，可使入如下命令：
```bash
mkdir build && cd build
cmake -DFASTDEPLOY_INSTALL_DIR=fastdeploy-xpu ..
make -j8
# 成功编译之后，会生成可运行 demo
```

5. 部署 ResNet50_vd 分类模型到昆仑芯 XPU 上，可使用如下命令：
```bash
cd FastDeploy/examples/vision/classification/paddleclas/xpu/cpp/build/
./infer_demo infer_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/204544266-11ae044b-d5ee-4613-a026-e1dec788bed7.png">
