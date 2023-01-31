English | [简体中文](README_CN.md)
# PaddleSeg Quantized Model Deployment
FastDeploy already supports the deployment of quantitative models and provides a tool to automatically compress model with just one click.
You can use the one-click automatical model compression tool to quantify and deploy the models, or directly download the quantified models provided by FastDeploy for deployment.

## FastDeploy One-Click Automation Model Compression Tool
FastDeploy provides an one-click automatical model compression tool that can quantify a model simply by entering configuration file. 
For details, please refer to [one-click automatical compression tool](../../../../../tools/common_tools/auto_compression/).
Note: The quantized classification model still needs the deploy.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.

## Download the Quantized PaddleSeg Model
You can also directly download the quantized models in the following table for deployment (click model name to download).

Note:
- Runtime latency is the inference latency of the model on various Runtimes, including CPU->GPU data copy, GPU inference, and GPU->CPU data copy time. It does not include the respective pre and post processing time of the models.
- The end-to-end latency is the latency of the model in the actual inference scenario, including the pre and post processing of the model.
- The measured latencies are averaged over 1000 inferences, in milliseconds.
- INT8 + FP16 is to enable the FP16 inference option for Runtime while inferring the INT8 quantization model.
- INT8 + FP16 + PM is the option to use Pinned Memory while inferring INT8 quantization model and turning on FP16, which can speed up the GPU->CPU data copy speed.
- The maximum speedup ratio is obtained by dividing the FP32 latency by the fastest INT8 inference latency.
- The strategy is quantitative distillation training, using a small number of unlabeled data sets to train the quantitative model, and verify the accuracy on the full validation set, INT8 accuracy does not represent the highest INT8 accuracy.
- The CPU is Intel(R) Xeon(R) Gold 6271C with a fixed CPU thread count of 1 in all tests. The GPU is Tesla T4, TensorRT version 8.4.15.

#### Runtime Benchmark
| Model                 |Inference Backends            | Hardware    | FP32 Runtime Latency   | INT8 Runtime Latency | INT8 + FP16 Runtime Latency  | INT8+FP16+PM Runtime Latency  |  Max Speedup    | FP32 mIoU | INT8 mIoU |  Method   |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar)  | Paddle Inference |    CPU    |     1138.04|   602.62 |None|None     |      1.89      |77.37 | 71.62 |Quantaware Distillation Training |

#### End to End Benchmark
| Model                 |Inference Backends             | Hardware    | FP32 End2End Latency   | INT8 End2End Latency | INT8 + FP16 End2End Latency  | INT8+FP16+PM End2End Latency  | Max Speedup   | FP32 mIoU | INT8 mIoU |   Method  |
| ------------------- | -----------------|-----------|  --------     |--------      |--------      | --------- |-------- |----- |----- |----- |
| [PP-LiteSeg-T(STDC1)-cityscapes](https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT_new.tar)  | Paddle Inference |    CPU    |     4726.65|   4134.91|None|None     |      1.14      |77.37 | 71.62 |Quantaware Distillation Training|

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
