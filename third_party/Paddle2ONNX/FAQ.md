## 常见问题

**Q1. Paddle模型转至ONNX模型过程中，提示『The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX』**  
A: 此提示为警告信息，模型仍然会正常进行转换。Paddle中`fluid.layers.multiclass_nms`算子中提供了`normalized`参数，用于表示输入box是否进行了归一化。而ONNX中的NMS算子只支持`normalized`参数为True的情况，当你转换的模型（一般是YOLOv3模型）中该参数为`False`的情况下，转换后的模型可能会与原模型存在diff。

**Q2. Paddle模型转至ONNX模型过程中，提示『Converting this model to ONNX need with static input shape, please fix input shape of this model』**  
A: 此提示为错误信息，表示该模型的转换需要固定的输入大小:
> 1. 模型来源于PaddleX导出，可以在导出的命令中，指定--fixed_input_shape=[Height,Width]，详情可见：[PaddleX模型导出文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/deploy/export_model.md)。
> 2. 模型来源于PaddleDetection导出，可以在导出模型的时候，指定 TestReader.inputs_def.image_shape=[Channel,Height,Width], 详情可见：[PaddleDetection模型导出文档](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md#设置导出模型的输入大小)。
> 3. 模型来源于自己构建，可在网络构建的`fluid.data(shape=[])`中，指定shape参数来固定模型的输入大小。
