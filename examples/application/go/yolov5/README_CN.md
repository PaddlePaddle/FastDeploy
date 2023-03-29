[English](README.md) | 简体中文
# YOLOv5 Golang 部署示例

本目录下提供`infer.go`, 使用CGO调用FastDeploy C API快速完成YOLOv5模型在CPU/GPU上部署的示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>1.0.4)或FastDeploy的Develop版本(x.x.x=0.0.0)
### 使用Golang和CGO工具进行YOLOv5模型推理部署

在当前目录下，下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
```bash
wget https://fastdeploy.bj.bcebos.com/dev/cpp/fastdeploy-linux-x64-0.0.0.tgz
tar xvf fastdeploy-linux-x64-0.0.0.tgz
```

将FastDeploy C API文件拷贝至当前目录
```bash
cp -r fastdeploy-linux-x64-0.0.0/include/fastdeploy_capi .
```

下载官方转换好的 YOLOv5 ONNX 模型文件和测试图片
```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

配置`infer.go`中的`cgo CFLAGS: -I`参数配置为C API文件路径，`cgo LDFLAGS: -L`参数配置为FastDeploy的动态库路径，动态库位于预编译库的`/lib`目录中
```bash
cgo CFLAGS: -I./fastdeploy_capi
cgo LDFLAGS: -L./fastdeploy-linux-x64-0.0.0/lib -lfastdeploy
```

将FastDeploy的库路径添加到环境变量
```bash
source /Path/to/fastdeploy-linux-x64-0.0.0/fastdeploy_init.sh 
```

编译Go文件`infer.go`
```bash
go build infer.go
```

编译完成后，使用如下命令执行可得到预测结果
```bash
# CPU推理
./infer -model yolov5s.onnx -image 000000014439.jpg -device 0
# GPU推理
./infer -model yolov5s.onnx -image 000000014439.jpg -device 1
```

可视化的检测结果图片保存在本地`vis_result.jpg`
