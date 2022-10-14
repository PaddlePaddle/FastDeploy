# PP-YOLOE-l量化模型 Python部署示例
本目录下提供的`infer.py`,可以帮助用户快速完成PP-YOLOE量化模型在CPU/GPU上的部署推理加速.

## 部署准备
### FastDeploy环境准备
- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/quick_start)

### 量化模型准备
- 1. 用户可以直接使用由FastDeploy提供的量化模型进行部署.
- 2. 用户可以使用FastDeploy提供的[一键模型量化工具](../../../../../../tools/quantization/),自行进行模型量化, 并使用产出的量化模型进行部署.(注意: 推理量化后的分类模型仍然需要FP32模型文件夹下的infer_cfg.yml文件, 自行量化的模型文件夹内不包含此yaml文件, 用户从FP32模型文件夹下复制此yaml文件到量化后的模型文件夹内即可.)


## 以量化后的PP-YOLOE-l模型为例, 进行部署
```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd /examples/vision/detection/paddledetection/quantize/python

#下载FastDeloy提供的ppyoloe_crn_l_300e_coco量化模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar
tar -xvf ppyoloe_crn_l_300e_coco_qat.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# 在CPU上使用ONNX Runtime推理量化模型
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco_qat --image 000000014439.jpg --device cpu --backend ort
# 在GPU上使用TensorRT推理量化模型
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco_qat --image 000000014439.jpg --device gpu --backend trt
```
