## Paddle Inference模型导出

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
python tools/export_model.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml --output_dir=./solov2_r50_fpn_1x_coco \
 -o weights=https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams

```

## ONNX模型导出

```bash
paddle2onnx --model_dir solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco.onnx \
            --enable_dev_version True
```
