# Yolov5 检测模型在 XPU 上的 C++ 部署示例

本目录下提供的 `infer.cc`，可以帮助用户快速完成 Yolov5 检测模型在 XPU 上的部署推理加速。

## 在 XPU 上部署 Yolov5 检测模型
请按照以下步骤完成在 XPU 上部署 Yolov5 检测模型：
1. 适用于 XPU 的 FastDeploy 库编译，具体请参考：[编译 FastDeploy](../../../../../../docs/cn/build_and_install/xpu.md)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-xpu/ FastDeploy/examples/vision/classification/paddleclas/xpu/cpp/
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s_infer.tar
tar -xvf yolov5s_infer.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

4. 编译部署示例，可使入如下命令：
```bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../fastdeploy-xpu/xpu.cmake -DFASTDEPLOY_INSTALL_DIR=fastdeploy-xpu ..
make -j8
# 成功编译之后，会生成可运行 demo
```

5. 部署 Yolov5 分类模型到 XPU 上，可使用如下命令：
```bash
cd FastDeploy/examples/vision/classification/paddleclas/xpu/cpp/build/
./infer_demo yolov5s_infer 000000014439.jpg
```

部署成功后运行结果如下：

<img width="640" src="https://user-images.githubusercontent.com/30516196/204545718-d259cf9c-00e5-49e3-b7bb-3a3be3db9fe3.png">
