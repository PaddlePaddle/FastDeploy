[English](README.md) | 简体中文
# PaddleDetection Rust 部署示例

本目录下提供`main.rs`和`build.rs`, 使用Rust的`bindgen`库调用FastDeploy C API快速完成PaddleDetection模型YOLOv8在CPU/GPU上部署的示例

在部署前，需确认以下三个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 3. 根据开发环境，使用Rustup安装[Rust](https://www.rust-lang.org/tools/install)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>1.0.4)或FastDeploy的Develop版本(x.x.x=0.0.0)
### 使用Rust和bindgen进行YOLOv8模型推理部署

在当前目录下，下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
```bash
wget https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz
tar xvf fastdeploy-linux-x64-0.0.0.tgz
```

下载官方转换好的 YOLOv8 ONNX 模型文件和测试图片
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov8s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

配置`build.rs`中的`cargo:rustc-link-search`参数配置为FastDeploy动态库路径，动态库位于预编译库的`/lib`目录中，`cargo:rustc-link-lib`参数配置为FastDeploy动态库`fastdeploy`，`headers_dir`变量配置为FastDeploy C API目录的路径
```bash
println!("cargo:rustc-link-search=./fastdeploy-linux-x64-0.0.0/lib");
println!("cargo:rustc-link-lib=fastdeploy");
let headers_dir = PathBuf::from("./fastdeploy-linux-x64-0.0.0/include");
```

将FastDeploy的库路径添加到环境变量
```bash
source /Path/to/fastdeploy-linux-x64-0.0.0/fastdeploy_init.sh 
```

使用Cargo编译Rust项目
```bash
cargo build
```

编译完成后，使用如下命令执行可得到预测结果
```bash
# CPU推理
cargo run -- --model yolov8s.onnx --image 000000014439.jpg --device 0
# GPU推理
cargo run -- --model yolov8s.onnx --image 000000014439.jpg --device 1
```

可视化的检测结果图片保存在本地`vis_result_yolov8.jpg`
