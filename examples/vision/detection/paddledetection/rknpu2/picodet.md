# Picodet RKNPU2模型转换文档

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

以转化RK3568的RKNN模型为例子，我们需要编辑tools/rknpu2/config/picodet_s_416_coco_lcnet_unquantized.yaml，来转换ONNX模型到RKNN模型。

**修改normalize参数**

如果你需要在NPU上执行normalize操作，请根据你的模型配置normalize参数，例如:

```yaml
mean:
  -
    - 127.5
    - 127.5
    - 127.5
std:
  -
    - 127.5
    - 127.5
    - 127.5
```

**修改outputs参数**
由于Paddle2ONNX版本的不同，转换模型的输出节点名称也有所不同，请使用[Netron](https://netron.app)对模型进行可视化，并找到以下蓝色方框标记的NonMaxSuppression节点，红色方框的节点名称即为目标名称。

例如，使用Netron可视化后，得到以下图片:

![](https://user-images.githubusercontent.com/58363586/212599781-e1952da7-6eae-4951-8ca7-bab7e6940692.png)

找到蓝色方框标记的NonMaxSuppression节点，可以看到红色方框标记的两个节点名称为p2o.Div.79和p2o.Concat.9,因此需要修改outputs参数，修改后如下:

```yaml
outputs_nodes: [ 'p2o.Div.79','p2o.Concat.9' ]
```

### 转换模型

```bash

# ONNX模型转RKNN模型
# 转换模型,模型将生成在picodet_s_320_coco_lcnet_non_postprocess目录下
python tools/rknpu2/export.py --config_path tools/rknpu2/config/picodet_s_416_coco_lcnet_unquantized.yaml \
                              --target_platform rk3588
```
