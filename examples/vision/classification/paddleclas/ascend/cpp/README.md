# PaddleClas Ascend NPU C++ 部署示例
本目录下提供的 `infer.cc`，可以帮助用户快速完成 PaddleClas 模型在华为昇腾NPU上的部署.
本例在鲲鹏920+Atlas 300I Pro的硬件平台下完成测试.

## 部署准备
### 华为昇腾NPU 部署环境编译准备
- 1. 软硬件环境满足要求，以及华为昇腾NPU的部署编译环境的准备，请参考：[FastDeploy 华为昇腾NPU部署环境编译准备](../../../../../../docs/cn/build_and_install/ascend.md)  

## 在 华为昇腾NPU 上部署ResNet50_Vd分类模型
请按照以下步骤完成在 华为昇腾NPU 上部署 ResNet50_Vd 模型：
1. 完成[华为昇腾NPU 部署环境编译准备](../../../../../../docs/cn/build_and_install/ascend.md)

2. 将编译后的库拷贝到当前目录，可使用如下命令：
```bash
cp -r FastDeploy/build/fastdeploy-cann/ FastDeploy/examples/vision/classification/paddleclas/cann/cpp/
```

3. 在当前路径下载部署所需的模型和示例图片：
```bash
mkdir models && mkdir images
# 下载模型,并放置于models目录下
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
cp -r ResNet50_vd_infer models
# 下载图片,放置于images目录下
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
cp -r ILSVRC2012_val_00000010.jpeg images
```

4. 编译部署示例，用户直接运行本目录下的`build.sh`文件,或者使用如下命令：
```bash
mkdir build && cd build
cmake cmake -DCMAKE_TOOLCHAIN_FILE=../fastdeploy-cann/cann.cmake -DFASTDEPLOY_INSTALL_DIR=fastdeploy-cann ..
make -j8
make install
# 成功编译之后，在build目录下, 会生成 install 文件夹，里面有一个运行 demo 和部署所需的库
```

5. 运行示例
用户直接运行`install`文件夹下的`run.sh`脚本即可.

部署成功后输出结果如下：
```bash
ClassifyResult(
label_ids: 153, 
scores: 0.685547, 
)
#此结果出现后,还会出现一些华为昇腾自带的log信息,属正常现象.
```