[English](README.md) | 简体中文
# YOLOv5 Java 部署示例

本目录下提供`java/InferDemo.java`, 使用`Java`调用`C++`API快速完成`PaddleDetection`模型`YOLOv5`在Linux上部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)



使用`Java`调用`C++` API 可以分为两步：

* 在`C++`端生成动态链接库。
* 在`Java`端调用动态链接库。

## C++端生成动态链接库
首先，切换路径到`cpp`目录，将`jdk`目录下的`jni.h`和`jni_md.h`拷贝到当前`cpp`目录下。
```shell
cp /PathJdk/jdk-17.0.6/include/jni.h ./
cp /Pathjdk/jdk-17.0.6/include/linux/jni_md.h ./
```

接着，在`cpp`目录下执行以下命令，进行编译，生成动态链接库。
> 注意：编译时需要通过`FASTDEPLOY_INSTALL_DIR`选项指明`FastDeploy`预编译库位置, 当然也可以是自己编译的`FastDeploy`库位置。
```shell
mkdir build && cd build
cmake .. -FASTDEPLOY_INSTALL_DIR /fast-deploy-path
make -j
```
编译成功后，动态链接库会存放在`cpp/build`目录下，`Linux`下以`.so`结尾，`Windows`下以`.dll`结尾。

## 使用JAVA调用动态链接库

将`FastDeploy`的库路径添加到环境变量，注意替换为自己的`FastDeploy`库所在路径。
```bash
source /Path/to/fastdeploy-linux-x64-1.0.4/fastdeploy_init.sh
```
下载YOLOv5模型文件和测试图片
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

编译Java文件
```shell
javac InferDemo.java
```

编译完成后，执行如下命令可得到预测结果，其中第一个参数指明下载的模型路径，第二个参数指明测试图片路径。
```shell
java InferDemo ./yolov5s.onnx ./000000014439.jpg
```
可视化的检测结果图片保存在本地`vis_result.jpg`。
