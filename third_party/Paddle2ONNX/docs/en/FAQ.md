# FAQ
[查看中文版](../zh/FAQ.md)
### Q1: What does the information "The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX" mean in converting a model?

A: This is a warning and model conversion will not be influenced. The operator fluid.layers.multiclass_nms in PaddlePaddle has a normalized parameter, representing if the iput box has done normalization, and if its value is False in your model(mostly Yolov3), the inference result may have diff with orignal model.

### Q2: What does the information "Converting this model to ONNX need with static input shape, please fix input shape of this model" mean in converting a model?

A: This implies an error, and the input shape of the model should be fixed:

- If the model is originated from PaddleX, you can designate it with --fixed_input_shape=[Height,Width]. Details please refer to [Exporting models in PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/export_model.md).
- If the model is originated from PaddleDetection, you can designate it with TestReader.inputs_def.image_shape=[Channel,Height,Width]. Details please refer to [Exporting models in PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/deploy/EXPORT_ONNX_MODEL.md).
- If the network of the model is built manually, you can designate it in fluid.data(shape=[]) by setting the shape parameter to fix the input size.

### Q4: Fixed shape is required, refer this doc for more information
- Sometimes the input shape of paddle model must be fixed for successful conversion due to the difference of operators between PaddlePaddle and ONNX。
- For a image classification model, input shape of `[-1, 3, -1, -1]` is not fixed while `[1, 3, 224, 224]` is fixed.
