# FastDeploy模型

目前支持的模型如下
- [fastdeploy.vision.ppcls.Model](vision/ppcls.md) PaddleClas里的所有分类模型
- [fastdeploy.vision.ultralytics/YOLOv5](vision/ultralytics.md) [ultralytics/yolov5](https://github.com/ultralytics/yolov5)模型

具体模型使用方式可参考各模型文档API和示例说明。 各模型在运行时均有默认的Runtime配置，本文档说明如何修改模型的后端配置，其中如下代码为跑YOLOv5的模型Python示例代码
```
import fastdeploy as fd
model = fd.vision.ulttralytics.YOLOv5("yolov5s.onnx")

import cv2
im = cv2.imread('bus.jpg')

result = model.predict(im)

print(model.runtime_option)
```
通过`print(model.runtime_option)`可以看到如下信息
```
RuntimeOption(
  backend : Backend.ORT                  # 当前推理后端为ONNXRuntime
  cpu_thread_num : 8                     # 推理时CPU线程数设置（仅当模型在CPU上推理时有效）
  device : Device.GPU                    # 当前推理设备为GPU
  device_id : 0                          # 当前推理设备id为0
  model_file : yolov5s.onnx              # 模型文件路径
  model_format : Frontend.ONNX           # 模型格式，当前为ONNX格式
  ort_execution_mode : -1                # ONNXRuntime后端的配置参数，-1表示默认
  ort_graph_opt_level : -1               # ONNXRuntime后端的配置参数, -1表示默认
  ort_inter_op_num_threads : -1          # ONNXRuntime后端的配置参数，-1表示默认
  params_file :                          # 参数文件（ONNX模型无此文件）
  trt_enable_fp16 : False                # TensorRT参数
  trt_enable_int8 : False                # TensorRT参数
  trt_fixed_shape : {}                   # TensorRT参数
  trt_max_batch_size : 32                # TensorRT参数
  trt_max_shape : {}                     # TensorRT参数
  trt_max_workspace_size : 1073741824    # TensorRT参数
  trt_min_shape : {}                     # TensorRT参数
  trt_opt_shape : {}                     # TensorRT参数
  trt_serialize_file :                   # TensorRT参数
)
```

会注意到参数名以`ort`开头的，均为ONNXRuntime后端专有的参数；以`trt`的则为TensorRT后端专有的参数。各后端与参数的配置，可参考[RuntimeOption](runtime_option.md)说明。

## 切换模型推理方式

一般而言，用户只需关注推理是在哪种Device下即可。 当然有更进一步需求，可以再为Device选择不同的Backend，但配置时注意Device与Backend的搭配。 如Backend::TRT只支持Device为GPU, 而Backend::ORT则同时支持CPU和GPU

```
import fastdeploy as fd
option = fd.RuntimeOption()
option.device = fd.Device.CPU
option.cpu_thread_num = 12
model = fd.vision.ulttralytics.YOLOv5("yolov5s.onnx", option)
print(model.runtime_option)
```
