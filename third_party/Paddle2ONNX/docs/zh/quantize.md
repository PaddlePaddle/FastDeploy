# PaddleSlim 量化模型导出 ONNX
PaddleSlim 有两种常用的量化方法，离线量化 (PTQ) 和量化训练 (QAT)，目前 Paddle2ONNX 已经支持将两种量化方法量化后的模型导出为 ONNX，并使用 ONNXRuntime 在 CPU 上或使用 TensorRT 8.X 版本在 GPU 上进行加速推理。

## 量化环境需求
1. PaddlePaddle > 2.3.2
2. Paddle2ONNX >= 1.0.0rc4
3. PaddleSlim >= 2.3.3

## 模型量化及导出注意事项
1. 使用 PaddleSlim 量化时请设置 onnx_format=True

2. 使用 Paddle2ONNX 导出 Paddle 量化模型为 ONNX 时，请根据部署的 backend 设置对应的 deploy_backend，示例如下：

```
# 使用 ONNXRuntime 在 CPU 上部署，导出成功后会生成量化模型 quant_model.onnx
paddle2onnx --model_dir ./ --model_filename model.pdmodel --params_filename model.pdiparams --save_file quant_model.onnx --opset_version 13 --enable_dev_version True --deploy_backend onnxruntime --enable_onnx_checker True

# 使用 TensorRT 在 GPU 上部署，导出成功后会生成 float_model.onnx 和 TensorRT 加载用的量化表 calibration.cache 文件
paddle2onnx --model_dir ./ --model_filename model.pdmodel --params_filename model.pdiparams --save_file float_model.onnx --opset_version 13 --enable_dev_version True --deploy_backend tensorrt --enable_onnx_checker True
```

3. 请确保 PaddleSlim 模型量化后，生成 3 个文件，分别是模型文件、权重文件和 scale 文件。PaddleSlim 量化 demo 和接口请查阅：[PaddleSlim 离线量化 demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)  
一个简单的量化配置说明如下：  

```
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle import fluid
place = fluid.CPUPlace()
exe = fluid.Executor(place)
ptq = PostTrainingQuantization(
    executor=exe,
    model_dir=model_path, # 待量化模型的存储路径
    sample_generator=val_reader, # 输入数据 reader
    batch_size=batch_size,
    batch_nums=batch_nums,
    algo=algo, # 量化算法支持 hist，KL，mse 等多种算法
    quantizable_op_type=quantizable_op_type,
    is_full_quantize=False, # 是否开启全量化
    optimize_model=False, # 量化前是否先对模型进行优化，如需导出为 ONNX 格式，请关闭此配置
    onnx_format=True, # 量化 OP 是否为 ONNX 格式，如需导出为 ONNX 格式，请将此配置打开
    skip_tensor_list=skip_tensor_list,
    is_use_cache_file=is_use_cache_file)
ptq.quantize() # 对模型进行量化
ptq.save_quantized_model(int8_model_path) # 保存量化后的模型，int8_model_path 为量化模型的保存路径
```

4. ONNXRuntime 部署量化模型和 float 模型的类似，无需特殊设置，TensorRT 部署量化模型参考：[TensorRT 部署示例](https://github.com/PaddlePaddle/Paddle2ONNX/tree/model_zoo/hardwares/tensorrt)

## FAQ

1. 模型导出时提示 fake_quantize_dequantize_*  或 fake_quantize_* 等 OP 不支持

答：使用 PaddleSlim 离线量化时没有开启 onnx_format 开关，请开启 onnx_format 开关之后重新导出量化模型。  

2. 量化模型使用 ONNXRuntime 在 CPU 端推理时精度相比 Paddle-TRT、MKLDNN 或者 Paddle 原生有明显下降  

答：如遇到使用 ONNXRuntime 推理时精度下降较多，可先用 PaddleInference 原生推理(不开启任何优化)验证是否量化模型精度本身就较低，如只有 ONNXRuntime 精度下降，请在终端执行：lscpu 命令，在 Flags 处查看机器是否支持 avx512-vnni 指令集，因为 ONNXRuntime 对量化模型推理还不是特别完备的原因，在不支持 avx512-vnni 的机器上可能会存在数据溢出的问题导致精度下降。  

可进一步使用如下脚本确认是否为该原因导致的精度下降，在使用 ONNXRuntime 推理时将优化全都关闭，然后再测试精度是否不再下降，如果还是存在精度下降问题，请提 ISSUE 给我们。

```
import onnxruntime as ort
providers = ['CPUExecutionProvider'] # 指定用 CPU 进行推理
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL # 关闭所有的优化
sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options) # model_path 为 ONNX 模型
pred_onnx = sess.run(None, input_dict) # 进行推理
```

3. CPU 端量化模型相比于非量化模型没有加速，反而变慢了  

答：模型量化相比非量化模型变慢了可以从以下几个原因分析：  

(1) 检查机器是否支持 avx2， avx512 或 avx512_vnni 指令，在支持 avx512_vnni 的 CPU 上精度和加速效果最好  

(2) 检查是否在 CPU 推理，当前导出的 ONNX 模型仅支持使用 ONNXRuntime 在 CPU 上进行推理加速  

(3) 量化模型对计算量大的 Conv 或 MatMul 等 OP 加速明显，如果模型中 Conv 或 MatMul 的计算量本身很小，那么量化可能并不会带来推理加速  

(4) 使用如下命令获得 ONNXRuntime 优化后的模型 optimize_model.onnx，然后使用 VisualDl 或 netron 等可视化工具可视化模型，检查以下两项：  

    1). 检查原模型中的 Conv、MatMul 和 Mul 等 OP 是否已经优化为 QLinearConv、QLinearMatMul 和  QLinearMul 等量化相关 OP  

    2). 检查优化后的模型中 QLinearConv 或 QLinearMatMul 等量化 OP 是否被 sigmod 或 Mean 非量化 OP 分开得很散，多个量化 OP 链接在一起，不需量化和反量化获得的加速效果最明显，如果是激活函数导致的 QLinearConv 等量化 OP 被分开，推荐将激活函数替换为 Relu 或 LeakyRelu 再进行测试

```
import onnxruntime as ort
providers = ['CPUExecutionProvider'] # 指定用 CPU 进行推理
sess_options = ort.SessionOptions()
sess_options.optimized_model_filepath = "./optimize_model.onnx" # 生成 ONNXRuntime 优化后的图，保存为 optimize_model.onnx
sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options) # model_path 为 ONNX 模型
```
