# ONNX 模型优化工具

## 1. ONNX 模型 Shape 推理

有时遇到拿到的 ONNX 模型缺少中间节点的 shape 信息，可以使用 `onnx_infer_shape` 来进行 shape 推理，此脚本源于[onnxruntime](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/symbolic_shape_infer.py)，使用方式如下
```
python onnx_infer_shape.py --input model.onnx --output new_model.onnx
```

## 2. 裁剪 ONNX 模型

在部分场景下，我们可能只需要整个模型的一部分，那么可以使用 `prune_onnx_model.py` 来裁剪模型，如我们只需要模型中的输出 `x` 和 `y` 及其之前的节点即可，那么可使用如下方式处理模型
```
python prune_onnx_model.py --model model.onnx --output_names x y --save_file new_model.onnx
```

其中 `output_names` 用于指定最终模型的输出 tensor，可以指定多个

## 3. 修改模型中间节点命名（包含输入、输出重命名）

```
python rename_onnx_model.py --model model.onnx --origin_names x y z --new_names x1 y1 z1 --save_file new_model.onnx
```

其中 `origin_names` 和 `new_names`，前者表示原模型中各个命名（可指定多个），后者表示新命名，两个参数指定的命名个数需要相同
