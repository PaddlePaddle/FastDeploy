[English](README.md) | 中文

# Intel GPU(独立显卡/集成显卡)的使用

FastDeploy通过OpenVINO后端支持Intel GPU显卡的使用。整体在部署模型时，与现有FastDeploy部署其它模型的流程类似，但在GPU上推理存在以下2个注意事项

- OpenVINO在显卡上推理时，要求模型的输入保持固定
- OpenVINO在显卡上支持的OP数量，与CPU不一致，需要异构执行

目前PaddleClas中所有OP均可使用GPU运行，而一些模型如PPYOLOE，则需要异构执行。具体使用示例可参考此目录下示例

## 输入固定说明

针对一个视觉模型的推理包含3个环节
- 输入图像，图像经过预处理，最终得到要输入给模型Runtime的Tensor
- 模型Runtime接收Tensor，进行推理，得到Runtime的输出Tensor
- 对Runtime的输出Tensor做后处理，得到最后的结构化信息，如DetectionResult, SegmentationResult等等

而输入固定，也即表示要求Runtime接收的Tensor，每次数据大小是一样的，不能变化。现有FastDeploy中，例如PP-OCR, RCNN这些每次输入给模型的大小就是在不断变化的（），因此暂不支持。而对于PaddleClas模型、PP-YOLOE、PicoDet，YOLOv5等，每次预处理后的数据大小是一样，则可以支持。

同时，我们在从框架导出部署模型时，可能也未进行Shape固定，例如PaddleClas的ResNet50模型，虽然推理时，一直接收的是[1, 3, 224, 224]大小的数据，但实际上导出模型时，输入的Shape被设定为了[-1, 3, -1, -1]，这也会导致OpenVINO无法确认模型的输入Shape。

FastDeploy提供如下接口，帮助来固定模型的Shape

- Python: `RuntimeOption.set_openvino_shape_info()`
- C++: `RuntimeOption::SetOpenVINOShapeInfo()`

## OP支持说明

深度学习模型本质是一个拓扑有向图，而图中的每一个节点，即为一个算子OP(Operator)。受限于不同推理引擎代码的实现，各后端支持的OP数量不一致。对于OpenVINO而言，在CPU和GPU上同样支持的OP数量不同，这也就意味着，同样一个模型使用OpenVINO可以跑在CPU上，但不一定能跑在GPU上。以PP-YOLOE为例，在GPU上直接跑，会出现如下提示，即表示`MulticlassNms`这个OP不被GPU支持。
```
RuntimeError: Operation: multiclass_nms3_0.tmp_1 of type MulticlassNms(op::v0) is not supported
```

这种情况下，我们可以通过异构的方式来执行模型，即让不支持的OP跑在CPU上，其余OP仍然在GPU上跑。

通过如下接口的设定，使用异构执行

### Python
```
import fastdeploy as fd
option = fd.RuntimeOption()
option.use_openvino_backend()
option.set_openvino_device("HETERO:GPU,CPU")
option.set_openvino_cpu_operators(["MulticlassNms"])
```

### C++
```
fastdeploy::RuntimeOption option;
option.UseOpenVINOBackend();
option.SetOpenVINODevice("HETERO:GPU,CPU");
option.SetOpenVINOCpuOperators({"MulticlassNms"});
```
