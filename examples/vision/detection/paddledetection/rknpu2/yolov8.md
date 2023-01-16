# YOLOv8 RKNPU2模型转换文档

以下步骤均在Ubuntu电脑上完成，请参考配置文档完成转换模型环境配置。下面以yolov8为例子,教大家如何转换PaddleDetection模型到RKNN模型。


### 导出ONNX模型

```bash
# 下载Paddle静态图模型并解压

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir yolov8_n_500e_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file yolov8_n_500e_coco/yolov8_n_500e_coco.onnx \
            --enable_dev_version True

# 固定shape
python -m paddle2onnx.optimize --input_model yolov8_n_500e_coco/yolov8_n_500e_coco.onnx \
                                --output_model yolov8_n_500e_coco/yolov8_n_500e_coco.onnx \
                                --input_shape_dict "{'image':[1,3,640,640]}"
```

### 编写模型导出配置文件
**修改outputs参数**
由于Paddle2ONNX版本的不同，转换模型的输出节点名称也有所不同，请使用[Netron](https://netron.app)对模型进行可视化，并找到以下蓝色方框标记的NonMaxSuppression节点，红色方框的节点名称即为目标名称。

例如，使用Netron可视化后，得到以下图片:

![](https://user-images.githubusercontent.com/58363586/212599658-8a2c4b79-f59a-40b5-ade7-f77c6fcfdf2a.png)

找到蓝色方框标记的NonMaxSuppression节点，可以看到红色方框标记的两个节点名称为tmp_17和p2o.Concat.9,因此需要修改outputs参数，修改后如下:

```yaml
outputs_nodes: [ 'p2o.Div.1','p2o.Concat.49' ]
```

### 转换模型

```bash

# ONNX模型转RKNN模型
# 转换模型,模型将生成在picodet_s_320_coco_lcnet_non_postprocess目录下
python tools/rknpu2/export.py --config_path tools/rknpu2/config/picodet_s_416_coco_lcnet_unquantized.yaml \
                              --target_platform rk3588
```
