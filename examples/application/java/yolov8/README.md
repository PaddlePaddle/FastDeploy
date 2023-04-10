English | [简体中文](README_CN.md)
# YOLOv8 Java Deployment Example

This directory provides examples that `java/InferDemo.java` uses `Java` to call FastDeploy `C++` API and finish the deployment of `YOLOv8` model。


Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2.  Download the precompiled deployment library and samples code according to your development environment. Refer to [FastDeploy Precompiled Library](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)


Using `Java` to call `C++` API can be divided into two steps:
* Generate dynamic link library in `C++` side.
* Call the dynamic link library in `Java` side.

## Generates dynamic link library
First, switch the path to the `cpp` directory and copy `jni.h` and `jni_md.h` which in `jdk` directory to current directory `cpp`.
```shell
cp /PathJdk/jdk-17.0.6/include/jni.h ./
cp /Pathjdk/jdk-17.0.6/include/linux/jni_md.h ./
```

Then, execute the following command in the `cpp` directory to compile and generate the dynamic link library.
> Note: you will need to specify the location of the FASTDEPLOY_INSTALL_DIR pre-compile library at compile time, but also the location of your own compiled FastDeploy library.
```shell
mkdir build && cd build
cmake .. -FASTDEPLOY_INSTALL_DIR /fast-deploy-path
make -j
```
After successful compilation, the dynamic link library will be stored in the `cpp/build` directory, ending in `.so` under `Linux` and `.dll` under `Windows`.

## Invoke dynamic link libraries using JAVA
Switch the path to the `java` directory and use the following command to add Fastdeploy library path to the environment variable. Note the path of the `FastDeploy` library replaced with your own.

```bash
source /Path/to/fastdeploy-linux-x64-1.0.4/fastdeploy_init.sh
```
Download the `YOLOv8` model file and test images.
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

Compiling Java files.
```shell
javac InferDemo.java
```

After compiling, run the following command to get the predicted result, where the first parameter indicates the path of the downloaded model and the second parameter indicates the path of the test image.
```shell
java InferDemo ./yolov8s.onnx ./000000014439.jpg
```
Then visualized inspection result is saved in the local image `vis_result.jpg`.
