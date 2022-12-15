English | [中文](README_CN.md)

# Deploy on Intel GPU

Intel GPU is supported by OpenVINO backend in FastDeploy. There're two notices while inference on Intel GPU

- The model's inputs shape have to be fixed
- There may exists some operators supported on CPU but not supported on GPU

FastDeploy provides two examples for these situations in this directory

## Fixed input shape

While deploying a compute vision model, it includes 3 steps
- Input a image data, after the preprocessing steps, we get the `tensors` which will be feed to the deeplearning model
- Inference the model by Runtime with the input `tensors`, and get the output `tensors`
- Postprocessing the output `tensors`, and get the final results we need, e.g `DetectionResult`, `SegmentationResult`

Fixed input shape means that the shape of the `tensors` received by the runtime is the same each time and cannot be changed. Such as PP-OCR and RCNN, the shape of each input to the model is changing, so it is not supported on Intel GPU temporarily. For PaddleClas model, PP-YOLOE, PicoDet, YOLOv5, etc., the input shape after each preprocessing is the same, which can be supported.

At the same time, when we export the deployment model from the framework, we may not have fixed the shape. For example, the ResNet50 model of PaddleClas receives [1, 3, 224, 224] size data all the time during reasoning, but actually when we export the model, the input shape is set to [- 1, 3, - 1, - 1], which also causes OpenVINO to be unable to confirm the input shape of the model.

FastDeploy provides the following interfaces to help fix the shape of the model

- Python: `RuntimeOption.set_openvino_shape_info()`
- C++: `RuntimeOption::SetOpenVINOShapeInfo()`

## Operators supporting

In essence, the deep learning model is a topological directed graph, and each node in the graph is an operator OP (Operator). Due to the implementation of different inference engine codes, the number of OPs supported by each backend is inconsistent. For OpenVINO, the number of OPs supported on the CPU and GPU is different, which means that the same model can run on the CPU, but may not be able to run on the GPU. Taking PP-YOLOE as an example, when running directly on the GPU, the following prompt will appear, which means that the 'MulticlassNms' OP is not supported by the GPU.

```
RuntimeError: Operation: multiclass_nms3_0.tmp_1 of type MulticlassNms(op::v0) is not supported
```

In this case, we can execute the model in a heterogeneous way, that is, let the unsupported OPs run on the CPU, and the remaining OPs still run on the GPU.

Heterogeneous execution is used through the settings of the following interfaces

**Python**

```
import fastdeploy as fd
option = fd.RuntimeOption()
option.use_openvino_backend()
option.set_openvino_device("HETERO:GPU,CPU")
option.set_openvino_cpu_operators(["MulticlassNms"])
```

**C++**

```
fastdeploy::RuntimeOption option;
option.UseOpenVINOBackend();
option.SetOpenVINODevice("HETERO:GPU,CPU");
option.SetOpenVINOCpuOperators({"MulticlassNms"});
```
