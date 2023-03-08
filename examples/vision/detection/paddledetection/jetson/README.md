English | [简体中文](README_CN.md)

# PaddleDetection Model Deployment

FastDeploy supports the SOLOV2 model of [PaddleDetection version 2.6](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.6).

You can enter the following command to get the static diagram model of SOLOV2.

```bash
# install PaddleDetection
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection

python tools/export_model.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml --output_dir=./inference_model \
 -o weights=https://paddledet.bj.bcebos.com/models/solov2_r50_fpn_1x_coco.pdparams
```

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
