# FastDeploy Runtime User Guideline

`Runtime`, the module for model inference in FastDeploy, currently integrates a variety of backends. It allows users to quickly complete inference in different model formats on different hardware, platforms and backends through a unified backend. This demo shows the inference on each hardware and backend.

## CPU Inference

Python demo

```python
import fastdeploy as fd
import numpy as np
option = fd.RuntimeOption()
# Set model path
option.set_model_path("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
# Use OpenVINO backend
option.use_openvino_backend()
# Initialize runtime
runtime = fd.Runtime(option)
# Get input info
input_name = runtime.get_input_info(0).name
# Constructing data for inference
results = runtime.infer({input_name: np.random.rand(1, 3, 224, 224).astype("float32")})
```

## GPU Inference

```python
import fastdeploy as fd
import numpy as np
option = fd.RuntimeOption()
# Set model path
option.set_model_path("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
# Use the GPU (0th GPU card)
option.use_gpu(0)
# Use Paddle Inference backend
option.use_paddle_backend()
# Initialize runtime
runtime = fd.Runtime(option)
# Get input info
input_name = runtime.get_input_info(0).name
# Constructing data for inference
results = runtime.infer({input_name: np.random.rand(1, 3, 224, 224).astype("float32")})
```

More Python/C++ inference demo, please refer to [FastDeploy/examples/runtime](../../examples/runtime)
