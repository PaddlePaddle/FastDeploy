# PaddleSeg Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/rknpu2.md) 

【注意】如你部署的为**PP-Matting**、**PP-HumanMatting**以及**ModNet**请参考[Matting模型部署](../../../matting)

本目录下提供`infer.py`快速完成PPHumanseg在RKNPU上部署的示例。执行如下脚本即可完成

```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

# 下载模型
wget https://bj.bcebos.com/fastdeploy/models/rknn2/human_pp_humansegv2_lite_192x192_pretrained_3588.tgz
tar xvf human_pp_humansegv2_lite_192x192_pretrained_3588.tgz

# 下载图片
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip images.zip

# 推理
python3 infer.py --model_file ./human_pp_humansegv2_lite_192x192_pretrained_3588/human_pp_humansegv2_lite_192x192_pretrained_3588.rknn \
                --config_file ./human_pp_humansegv2_lite_192x192_pretrained_3588/deploy.yaml \
                --image images/portrait_heng.jpg
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>


## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，
需要先调用DisableNormalizePermute(C++)或`disable_normalize_permute(Python)，在预处理阶段禁用归一化以及数据格式的转换。
## 其它文档

- [PaddleSeg 模型介绍](..)
- [PaddleSeg C++部署](../cpp)
- [模型预测结果说明](../../../../../../docs/api/vision_results/)
- [转换PPSeg RKNN模型文档](../README.md)
