[English](README.md) | 简体中文

# PaddleDetection TVM部署示例

在TVM上已经通过测试的PaddleDetection模型如下:

* picodet
* PPYOLOE

### Paddle模型转换为TVM模型

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX/tools/paddle
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_plus_crn_m_80e_coco.tgz
tar xvf ppyoloe_plus_crn_m_80e_coco.tgz
python prune_paddle_model.py --model_dir ppyoloe_plus_crn_m_80e_coco  \
                          --model_filename model.pdmodel \
                          --params_filename model.pdiparams \
                          --output_names  tmp_17 concat_14.tmp_0 \
                          --save_dir ppyoloe_plus_crn_m_80e_coco
python path/to/FastDeploy/tools/tvm/convert_model.py --model_path=./ppyoloe_plus_crn_m_80e_coco/model \
                       --shape_dict="{'image': [1, 3, 640, 640], 'scale_factor': [1, 2]}"
cp ppyoloe_plus_crn_m_80e_coco/infer_cfg.yml tvm_save
```
