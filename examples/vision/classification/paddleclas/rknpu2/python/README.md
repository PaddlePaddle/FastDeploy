# ResNet50_vd模型部署

## 转换模型
下面以 ResNet50_vd为例子，教大家如何转换分类模型到RKNN模型。

```bash
# 安装 paddle2onnx
pip install paddle2onnx

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir ResNet50_vd_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file ResNet50_vd_infer/ResNet50_vd_infer.onnx --enable_dev_version True --opset_version 12 --enable_onnx_checker True

# 固定shape，注意这里的inputs得对应netron.app展示的 inputs 的 name，有可能是image 或者 x
python -m paddle2onnx.optimize --input_model ResNet50_vd_infer/ResNet50_vd_infer.onnx --output_model ResNet50_vd_infer/ResNet50_vd_infer.onnx  --input_shape_dict "{'inputs':[1,3,224,224]}"

# 配置文件 ResNet50_vd_infer_rknn.yaml 如下:
model_path: ./ResNet50_vd_infer.onnx
output_folder: ./
target_platform: RK3588
normalize:
  mean: [[0, 0, 0]]
  std: [[1, 1, 1]]
outputs: []
outputs_nodes: []
do_quantization: False
dataset: 


# ONNX模型转RKNN模型
python tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/ResNet50_vd_infer_rknn.yaml \
        --target_platform rk3588
```

## 执行代码

```bash
python3 infer.py --model_file ./ResNet50_vd_infer/ResNet50_vd_infer_rk3588.rknn  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# 运行完成后返回结果如下所示
ClassifyResult(
label_ids: 153, 
scores: 0.684570, 
)
```