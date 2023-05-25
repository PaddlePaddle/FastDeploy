[English](README.md) | 简体中文

# PaddleDetection TVM部署示例

在TVM上已经通过测试的PaddleDetection模型如下:

* picodet
* PPYOLOE

### Paddle模型转换为TVM模型

由于TVM不支持NMS算子，因此在转换模型前我们需要对PaddleDetection模型进行裁剪，将模型的输出节点改为NMS节点的输入节点。
输入以下命令，你将得到一个裁剪后的PPYOLOE模型。

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
```

裁剪完模型后我们就可以通过tvm python库实现编译模型，这里为了方便大家使用，提供了转换脚本。
输入以下命令，你将得到转换过后的TVM模型。
注意，FastDeploy在推理PPYOLOE时不关依赖模型，还依赖yml文件，因此你还需要将对应的yml文件拷贝到模型目录下。

```bash
python path/to/FastDeploy/tools/tvm/paddle2tvm.py --model_path=./ppyoloe_plus_crn_m_80e_coco/model \
                       --shape_dict="{'image': [1, 3, 640, 640], 'scale_factor': [1, 2]}"
cp ppyoloe_plus_crn_m_80e_coco/infer_cfg.yml tvm_save
```
