# PaddleDetection RKNPU2部署示例

## 支持模型列表

目前FastDeploy支持如下模型的部署
- [PicoDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)

## 准备PaddleDetection部署模型以及转换模型
RKNPU部署模型前需要将Paddle模型转换成RKNN模型，具体步骤如下:
* Paddle动态图模型转换为ONNX模型，请参考[PaddleDetection导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)
  ,注意在转换时请设置**export.nms=True**.
* ONNX模型转换RKNN模型的过程，请参考[转换文档](../../../../../docs/cn/faq/rknpu2/export.md)进行转换。


## 模型转换example
下面以Picodet-npu为例子,教大家如何转换PaddleDetection模型到RKNN模型。
```bash
# 下载Paddle静态图模型并解压
wget https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_416_coco_lcnet.tar
tar xvf picodet_s_416_coco_lcnet.zip

# 静态图转ONNX模型，注意，这里的save_file请和压缩包名对齐
paddle2onnx --model_dir picodet_s_416_coco_lcnet \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
            --enable_dev_version True

python -m paddle2onnx.optimize --input_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --output_model picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx \
                                --input_shape_dict "{'image':[1,3,416,416]}"
# ONNX模型转RKNN模型
# 转换模型,模型将生成在picodet_s_320_coco_lcnet_non_postprocess目录下
python tools/rknpu2/export.py --config_path tools/rknpu2/config/RK3588/picodet_s_416_coco_lcnet.yaml
```

- [Python部署](./python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
