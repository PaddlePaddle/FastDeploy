# Paddle2Caffe

## 简介

Paddle2Caffe支持将**PaddlePaddle**模型格式转化到**Caffe**模型格式。通过Caffe可以完成将Paddle模型后续到多种硬件的部署，包括TensorRT/OpenVINO/DDK/SNPE等等。
Paddle2Caffe目前为[Contrib]预览版本

## 环境依赖

- 详见requirements.txt

## 使用

### 获取PaddlePaddle部署模型

Paddle2Caffe在导出模型时，需要传入部署模型格式，包括两个文件
- `model_name.pdmodel`: 表示模型结构
- `model_name.pdiparams`: 表示模型参数
  [注意] 这里需要注意，两个文件其中参数文件后辍为`.pdiparams`，如你的参数文件后辍是`.pdparams`，那说明你的参数是训练过程中保存的，当前还不是部署模型格式。 部署模型的导出可以参照各个模型套件的导出模型文档。

或者使用PaddlePaddle<2版本产出的模型命名格式
- `__model__`: 表示模型结构
- `__params__`: 表示模型参数

### 命令行转换

```
paddle2caffe --model_dir saved_inference_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams \
             --save_file model2caffe \
             --enable_caffe_custom False
```
如你参考文档[CaffePlugin](CaffePlugin.md)进行了第三方caffe算子的扩充，推荐使用`--enable_caffe_custom True`，从而获得对SSD结构、YOLO结构的支持  
具体详见参考文档


#### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 配置包含Paddle模型的目录路径|
|--model_filename |**[可选]** 配置位于`--model_dir`下存储网络结构的文件名|
|--params_filename |**[可选]** 配置位于`--model_dir`下存储模型参数的文件名称|
|--save_file | 指定转换后的模型保存目录以及文件前缀 |
|--enable_caffe_custom | **[可选]** 是否使用第三方扩充的算子（例如priorbox、upsample等），默认为False |
|--version |**[可选]** 查看paddle2caffe版本 |


## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
