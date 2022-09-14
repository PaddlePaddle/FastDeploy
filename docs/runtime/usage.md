# FastDeploy Runtime使用文档

`Runtime`作为FastDeploy中模型推理的模块，目前集成了多种后端，用户通过统一的后端即可快速完成不同格式的模型，在各硬件、平台、后端上的推理。本文档通过如下示例展示各硬件、后端上的推理

## CPU推理

Python示例

```
import fastdeploy as fd
import numpy as np
option = fd.RuntimeOption()
# 设定模型路径
option.set_model_path("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
# 使用OpenVINO后端
option.use_openvino_backend()
# 初始化runtime
runtime = fd.Runtime(option)
# 获取输入名
input_name = runtime.get_input_info(0).name
# 构造数据进行推理
results = runtime.infer({input_name: np.random.rand(1, 3, 224, 224).astype("float32")})
```

## GPU推理
```
import fastdeploy as fd
import numpy as np
option = fd.RuntimeOption()
# 设定模型路径
option.set_model_path("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
# 使用GPU，并且使用第0张GPU卡
option.use_gpu(0)
# 使用Paddle Inference后端
option.use_openvino_backend()
# 初始化runtime
runtime = fd.Runtime(option)
# 获取输入名
input_name = runtime.get_input_info(0).name
# 构造数据进行推理
results = runtime.infer({input_name: np.random.rand(1, 3, 224, 224).astype("float32")})
```

更多Python/C++推理示例请直接参考[FastDeploy/examples/runtime](../../examples/runtime)
