[English](README.md) | 简体中文
# PaddleDetection RKNPU2部署示例

## 支持模型列表

目前FastDeploy支持如下模型的部署
- [PicoDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)

## 准备PaddleDetection部署模型以及转换模型
RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:
* Paddle动态图模型转换为ONNX模型，请参考[PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)
  ,注意在转换时请设置**export.nms=True**.
* ONNX模型转换RKNN模型的过程，请参考[转换文档](../../../../../docs/cn/faq/rknpu2/export.md)进行转换。


## 模型转换example
以下步骤均在Ubuntu电脑上完成，请参考配置文档完成转换模型环境配置。下面以Picodet-s为例子,教大家如何转换PaddleDetection模型到RKNN模型。

### 导出ONNX模型
```bash
# 下载Paddle静态图模型并解压
wget https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar
tar xvf picodet_s_416_coco_lcnet.tar

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir picodet_s_416_coco_lcnet \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
            --enable_dev_version True

# 固定shape
python -m paddle2onnx.optimize --input_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --output_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --input_shape_dict "{'image':[1,3,416,416]}"
```

### 编写模型导出配置文件
以转化RK3568的RKNN模型为例子，我们需要编辑tools/rknpu2/config/RK3568/picodet_s_416_coco_lcnet.yaml，来转换ONNX模型到RKNN模型。

**修改normalize参数**

如果你需要在NPU上执行normalize操作，请根据你的模型配置normalize参数，例如:
```yaml
model_path: ./picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx
output_folder: ./picodet_s_416_coco_lcnet
target_platform: RK3568
normalize:
  mean: [[0.485,0.456,0.406]]
  std: [[0.229,0.224,0.225]]
outputs: ['tmp_17','p2o.Concat.9']
```

**修改outputs参数**
由于Paddle2ONNX版本的不同，转换模型的输出节点名称也有所不同，请使用[Netron](https://netron.app)，并找到以下蓝色方框标记的NonMaxSuppression节点，红色方框的节点名称即为目标名称。

例如，使用Netron可视化后，得到以下图片:
![](https://user-images.githubusercontent.com/58363586/202728663-4af0b843-d012-4aeb-8a66-626b7b87ca69.png)

找到蓝色方框标记的NonMaxSuppression节点，可以看到红色方框标记的两个节点名称为tmp_17和p2o.Concat.9,因此需要修改outputs参数，修改后如下:
```yaml
model_path: ./picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx
output_folder: ./picodet_s_416_coco_lcnet
target_platform: RK3568
normalize: None
outputs: ['tmp_17','p2o.Concat.9']
```

### 转换模型
```bash

# ONNX模型转RKNN模型
# 转换模型,模型将生成在picodet_s_320_coco_lcnet_non_postprocess目录下
python tools/rknpu2/export.py --config_path tools/rknpu2/config/picodet_s_416_coco_lcnet.yaml \
                              --target_platform rk3588
```

### 修改模型运行时的配置文件

配置文件中，我们只需要修改**Preprocess**下的**Normalize**和**Permute**.

**删除Permute**

RKNPU只支持NHWC的输入格式，因此需要删除Permute操作.删除后，配置文件Precess部分后如下:
```yaml
Preprocess:
- interp: 2
  keep_ratio: false
  target_size:
  - 416
  - 416
  type: Resize
- is_scale: true
  mean:
  - 0.485
  - 0.456
  - 0.406
  std:
  - 0.229
  - 0.224
  - 0.225
  type: NormalizeImage
```

**根据模型转换文件决定是否删除Normalize**

RKNPU支持使用NPU进行Normalize操作，如果你在导出模型时配置了Normalize参数，请删除**Normalize**.删除后配置文件Precess部分如下:
```yaml
Preprocess:
- interp: 2
  keep_ratio: false
  target_size:
  - 416
  - 416
  type: Resize
```

## 其他链接
- [Cpp部署](./cpp)
- [Python部署](./python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
