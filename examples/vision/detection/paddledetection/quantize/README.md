English | [简体中文](README_CN.md)
# PaddleDetection Quantification Model Deployment
FastDeploy supports the deployment of quantification models and provides a convenient tool for automatic model compression. 
Users can use it to deploy models after quantification or directly deploy quantized models provided by FastDeploy.

## FastDeploy one-click model auto-compression tool
FastDeploy provides a one-click auto-compression tool that allows users to quantize models by simply entering a configuration file.  
Refer to [one-click auto-compression tool](../../../../../tools/common_tools/auto_compression/) for details. 

## Download the quantized PP-YOLOE-l model
Users can also directly download the quantized models in the table below. (Click the model name to download it)


Benchmark  table description:
- Runtime latency: model’s inference latency on multiple Runtimes, including CPU->GPU data copy, GPU inference, and GPU->CPU data copy time. It does not include the pre and post processing time of the model.
- End2End latency: model’s latency in the actual inference scenario, including the pre and post processing time of the model.
- Measured latency: The average latency after 1000 times of inference in milliseconds.
- INT8 + FP16: Enable FP16 inference for Runtime while inferring the INT8 quantification model
- INT8 + FP16 + PM: Use Pinned Memory to speed up the GPU->CPU data copy while inferring the INT8 quantization model with FP16 turned on.
- Maximum speedup ratio: Obtained by dividing the FP32 latency by the highest INT8 inference latency.
- The strategy is to use a few unlabeled data sets to train the model for quantification and to verify the accuracy on the full validation set. The INT8 accuracy does not represent the highest value.
- The CPU is Intel(R) Xeon(R) Gold 6271C, , and the number of CPU threads is fixed to 1. The GPU is Tesla T4 with TensorRT version 8.4.15.


#### Runtime Benchmark
| Model                 |Inference Backend            |Deployment Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  | Maximum Speedup Ratio    | FP32 mAP | INT8 mAP | Quantification Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | TensorRT         |    GPU    |  27.90 | 6.39 |6.44|5.95    |      4.67       | 51.4  | 50.7 | Quantized distillation training |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | Paddle-TensorRT |    GPU    |  30.89     |None  |  13.78 |14.01    |      2.24       | 51.4  | 50.5 | Quantized distillation training |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar)  | ONNX Runtime |    CPU    |     1057.82 |   449.52 |None|None    |      2.35        |51.4 | 50.0 | Quantized distillation training |

NOTE:
- The reason why TensorRT is faster than Paddle-TensorRT is that the multiclass_nms3 operator is removed during runtime

#### End2End Benchmark
| Model                 | Inference Backend            |Deployment Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency | INT8+FP16+PM End2End Latency  | Maximum Speedup Ratio    | FP32 mAP | INT8 mAP | Quantification Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | TensorRT         |    GPU    |  35.75 | 15.42 |20.70|20.85  |      2.32      | 51.4  | 50.7 | Quantized distillation training |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar )  | Paddle-TensorRT |    GPU    | 33.48    |None  |  18.47 |18.03   |     1.81       | 51.4  | 50.5 | Quantized distillation training |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar)  | ONNX Runtime |    CPU    |     1067.17 |   461.037 |None|None    |      2.31        |51.4 | 50.0 | Quantized distillation training |


## Detailed Deployment Tutorials  

- [Python Deployment](python)
- [C++ Deployment](cpp)
