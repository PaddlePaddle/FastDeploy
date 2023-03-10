# OpenVINO部署PaddleDetection模型

[English Version of this document](./openvino_ppdet_en.md)

此篇文档将帮助大家解决如何通过Paddle2ONNX将目标检测模型转为ONNX，再从ONNX转为OpenVINO IR并推理。

## 问题分析

- 1. OpenVINO目前在推理过程中，要求模型的各计算节点shape固定
- 2. 目标检测模型的后处理NMS节点中, 最终出来的目标数量是不确定的，即shape不固定

## 解决方案

针对上面的问题，我们在本目录下，提供了一个NMS转换插件，使得最终NMS出来的目标数量稳定不变。

- 因此最终得到的检测测果（shape为N*6），其中会包含部分无效目标，其label_id为-1，我们得到结果后过滤即可, 可以参考目录openvino_ppdet/yolov3_infer.py中的postprocess函数实现

## 模型分析

目前本文档中仅支持了PaddleDetection中的YOLOv3系列，包括YOLOv3-DarkNet, YOLOv3-Resnet34等等。PaddleDetection中的目标检测模型均包括3个输入值和2个输出值。3个输入值为`image`, `im_shape`, `scale_factor`，分别表示预处理后的图像数据N*3*H*W，图像原大小信息N*2, 图像缩放系数N*2，后面两个输入信息最终目的是要将检测框坐标还原到原图对应的位置。  

注意，为了支持OpenVINO，我们需要将YOLOv3的输入大小固定，例如3个输入形状固定为状[1,3,608,608], [1,2], [1,2]，即将batch_size，以及输入图像的高和宽都固定，其中batch_size必须为1，而高和宽为32的倍数即可，如[1,3,608,608]或[1,3,320,320]

## 转换过程

### 1. PaddleDetection导出模型

```
cd PaddleDetection
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
                             -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams \
                             --output_dir inference_model
# 假设模型最终保存在了/User/XXX/PaddleDetection/inference_model/yolov3_darknet53_270e_coco下
```

### 2. 模型导出为ONNX格式
首先源码安装Paddle2ONNX
```
# 如果事先已安装paddle2onnx，需先卸载
# pip uninstall paddle2onnx
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git checkout release/0.9
python setup.py install

cd experimental
```

当前你处于`Paddle2ONNX/experimental`目录，可以看到有个子目录`openvino_ppdet`，在当前目录下使用Python执行如下代码
```
import paddle2onnx
import paddle
from openvino_ppdet import nms_mapper
# 通过上面的`nms_mapper`的import来启用插件，替换了paddle2onnx原始的nms_mapper

model_prefix = "/User/XXX/PaddleDetection/inference_model/yolov3_darknet53_270e_coco/model"
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image": [1, 3, 608, 608],
    "scale_factor": [1, 2],
    "im_shape": [1, 2]
    }
onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict, opset_version=11)

with open("./yolov3.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```
确保执行过程中，输出了如下日志提示，这表示你成功使用了OpenVINO的NMS转换插件
```
===============================


You are using a nms convertor for OpenVINO!


===============================
```

## 3. 转为OpenVINO IR
如果你已经安装好OpenVINO，并已经初始化环境，这时可以使用mo.py将ONNX模型转为其IR
```
mo.py --framework onnx --input_model yolov3.onnx --output_dir ov_model
```

## 4. OpenVINO推理
当前你需要处于`Paddle2ONNX/experimental`目录下，可以看到有子目录`openvino_ppdet`，我们在这个目录下实现了一个简单的`yolov3_infer.py`，来帮助你快速上手推理，在此目录下执行如下示例代码即可
```
import sys
from openvino_ppdet.yolov3_infer import YOLOv3

xml_file = "ov_model/yolov3.xml"
bin_file = "ov_model/yolov3.bin"
model = YOLOv3(xml_file=xml_file,
               bin_file=bin_file,
               model_input_shape=[608, 608])
boxes = model.predict("./test.jpg", visualize_out="./result.jpg", threshold=0.5)
```
- model_input_shape需与上面导出ONNX时的shape一致，此参数会用于图像的预处理
- visualize_out为可视化结果保存路径，如不需要可视化，设为None即可
- threshold用于过滤结果，置信度低于threshold的结果将被过滤
