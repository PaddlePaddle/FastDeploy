# PaddleClas 模型RKNPU2部署

## 转换模型
下面以 ResNet50_vd为例子，教大家如何转换分类模型到RKNN模型。

```bash
# 安装 paddle2onnx
pip install paddle2onnx

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir ResNet50_vd_infer  \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams  \
            --save_file ResNet50_vd_infer/ResNet50_vd_infer.onnx  \
            --enable_dev_version True  \
            --opset_version 12  \
            --enable_onnx_checker True

# 固定shape，注意这里的inputs得对应netron.app展示的 inputs 的 name，有可能是image 或者 x
python -m paddle2onnx.optimize --input_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --output_model ResNet50_vd_infer/ResNet50_vd_infer.onnx \
                               --input_shape_dict "{'inputs':[1,3,224,224]}"
```                               

 ### 编写模型导出配置文件
以转化RK3588的RKNN模型为例子，我们需要编辑tools/rknpu2/config/ResNet50_vd_infer_rknn.yaml，来转换ONNX模型到RKNN模型。                              

默认的 mean=0, std=1是在内存做normalize，如果你需要在NPU上执行normalize操作，请根据你的模型配置normalize参数，例如:
```yaml
model_path: ./ResNet50_vd_infer.onnx
output_folder: ./
target_platform: RK3588
normalize:
  mean: [[0.485,0.456,0.406]]
  std: [[0.229,0.224,0.225]]
outputs: []
outputs_nodes: []
do_quantization: False
dataset: 
```


# ONNX模型转RKNN模型
```shell
python tools/rknpu2/export.py \
        --config_path tools/rknpu2/config/ResNet50_vd_infer_rknn.yaml \
        --target_platform rk3588
```

## 其他链接
- [Cpp部署](./cpp)
- [Python部署](./python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)