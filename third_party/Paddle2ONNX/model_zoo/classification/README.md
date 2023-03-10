# 图像分类模型库

本文档中模型库均来源于PaddleCls [release/2.3分支](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/)，在下表中提供了部分已经转换好的模型，如有更多模型或自行模型训练导出需求，可参见[ImageNet 预训练模型库
](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md)

其中类别id与明文标签对应关系参考文件[ImageNet标签](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/utils/imagenet1k_label_list.txt)

|模型名称|模型大小|下载地址|说明|
| --- | --- | --- | ---- |
|ResNet50|98M|[Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/resnet50.tar.gz) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/resnet50.onnx)| 使用ImageNet数据作为训练数据，1000个分类 |
|PPLCNet|11M|[Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/pplcnet.tar.gz) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/pplcnet.onnx)| 使用ImageNet数据作为训练数据，1000个分类 |
| MobileNetV3_small | 11M    | [Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.onnx) | 使用ImageNet数据作为训练数据，1000个分类 |
| EfficientNetB0_small | 11M    | [Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/efficientnetb0.tar.gz) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/efficientnetb0.onnx) | 使用ImageNet数据作为训练数据，1000个分类 |



## ONNX模型推理示例

各模型的推理前后处理参考本目录下的`infer.py`，以MobileNetV2为例，如下命令即可得到推理结果

```
# 安装onnxruntime
pip install onnxruntime
# 下载mobilenetv3模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.onnx
python infer.py --model mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg
```

你也可以使用Paddle框架进行推理验证

```
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
tar xvf mobilenetv3.tar.gz
python infer.py --model mobilenetv3 --image_path images/ILSVRC2012_val_00000010.jpeg --use_paddle_predict True
```
输出结果如下所示
```
TopK Indices:  [265 153 850 332 283]
TopK Scores:  [0.4966848  0.25181034 0.15389322 0.01496286 0.01342606]
```
分别表示该图识别结果，打分最高的前5个类别id，以及其相应的置信度分值。各类别id与明文标签参考[ImageNet标签](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/utils/imagenet1k_label_list.txt)
