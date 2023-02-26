[English](README.md) | 简体中文
# PaddleDetection模型部署

FastDeploy支持[PaddleDetection 2.6](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6)版本的SOLOv2模型，
你可以输入以下命令得到SOLOv2的动态图模型。

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6
python tools/export_model.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml --output_dir=./inference_model \
 -o weights=https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams
```

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
