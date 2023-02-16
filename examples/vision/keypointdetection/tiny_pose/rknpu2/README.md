[English](README.md) | 简体中文
# PP-TinyPose RKNPU2部署示例

## 模型版本说明

- [PaddleDetection release/2.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5)

目前FastDeploy支持如下模型的部署

- [PP-TinyPose系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/keypoint/tiny_pose/README.md)

## 准备PP-TinyPose部署模型

PP-TinyPose模型导出，请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)

**注意**:PP-TinyPose导出的模型包含`model.pdmodel`、`model.pdiparams`和`infer_cfg.yml`三个文件，FastDeploy会从yaml文件中获取模型在推理时需要的预处理信息。

## 模型转换example

### Paddle模型转换为ONNX模型

由于Rockchip提供的rknn-toolkit2工具暂时不支持Paddle模型直接导出为RKNN模型，因此需要先将Paddle模型导出为ONNX模型，再将ONNX模型转为RKNN模型。

```bash
# 下载Paddle静态图模型并解压
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir PP_TinyPose_256x192_infer \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file PP_TinyPose_256x192_infer/PP_TinyPose_256x192_infer.onnx \
            --enable_dev_version True

# 固定shape
python -m paddle2onnx.optimize --input_model PP_TinyPose_256x192_infer/PP_TinyPose_256x192_infer.onnx \
                                --output_model PP_TinyPose_256x192_infer/PP_TinyPose_256x192_infer.onnx \
                                --input_shape_dict "{'image':[1,3,256,192]}"
```

### ONNX模型转RKNN模型

为了方便大家使用，我们提供了python脚本，通过我们预配置的config文件，你将能够快速地转换ONNX模型到RKNN模型

```bash
python tools/rknpu2/export.py --config_path tools/rknpu2/config/PP_TinyPose_256x192_unquantized.yaml \
                              --target_platform rk3588
```

## 详细部署文档

- [模型详细介绍](../README_CN.md)
- [Python部署](./python)
- [C++部署](./cpp)