[English](README.md) | 简体中文
# YOLOv5 SOPHGO部署示例

## 支持模型列表

YOLOv5 v6.0部署模型实现来自[YOLOv5](https://github.com/ultralytics/yolov5/tree/v6.0),和[基于COCO的预训练模型](https://github.com/ultralytics/yolov5/releases/tag/v6.0)

## 准备YOLOv5部署模型以及转换模型

SOPHGO-TPU部署模型前需要将Paddle模型转换成bmodel模型，具体步骤如下:
- 下载预训练ONNX模型，请参考[YOLOv5准备部署模型](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/yolov5)
- ONNX模型转换bmodel模型的过程，请参考[TPU-MLIR](https://github.com/sophgo/tpu-mlir)

## 模型转换example

下面以YOLOv5s为例子,教大家如何转换ONNX模型到SOPHGO-TPU模型

## 下载YOLOv5s模型

### 下载ONNX YOLOv5s静态图模型
```shell
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s.onnx

```
### 导出bmodel模型

以转化BM1684x的bmodel模型为例子，我们需要下载[TPU-MLIR](https://github.com/sophgo/tpu-mlir)工程，安装过程具体参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。
### 1.	安装
``` shell
docker pull sophgo/tpuc_dev:latest

# myname1234是一个示例，也可以设置其他名字
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest

source ./envsetup.sh
./build.sh
```

### 2.	ONNX模型转换为bmodel模型
``` shell
mkdir YOLOv5s && cd YOLOv5s

# 在该文件中放入测试图片，同时将上一步下载的yolov5s.onnx放入该文件夹中
cp -rf ${REGRESSION_PATH}/dataset/COCO2017 .
cp -rf ${REGRESSION_PATH}/image .
# 放入onnx模型文件yolov5s.onnx

mkdir workspace && cd workspace

# 将ONNX模型转换为mlir模型，其中参数--output_names可以通过NETRON查看
model_transform.py \
    --model_name yolov5s \
    --model_def ../yolov5s.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --output_names output,350,498,646 \
    --test_input ../image/dog.jpg \
    --test_result yolov5s_top_outputs.npz \
    --mlir yolov5s.mlir

# 将mlir模型转换为BM1684x的F32 bmodel模型
model_deploy.py \
  --mlir yolov5s.mlir \
  --quantize F32 \
  --chip bm1684x \
  --test_input yolov5s_in_f32.npz \
  --test_reference yolov5s_top_outputs.npz \
  --model yolov5s_1684x_f32.bmodel
```
最终获得可以在BM1684x上能够运行的bmodel模型yolov5s_1684x_f32.bmodel。如果需要进一步对模型进行加速，可以将ONNX模型转换为INT8 bmodel，具体步骤参见[TPU-MLIR文档](https://github.com/sophgo/tpu-mlir/blob/master/README.md)。

## 其他链接
- [Cpp部署](./cpp)
