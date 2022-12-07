# PaddleClas 华为昇腾 NPU Python 部署示例
本目录下提供的 `infer.py`，可以帮助用户快速完成 PaddleClas 模型在华为昇腾NPU上的部署.
本例在鲲鹏920+Atlas 300I Pro的硬件平台下完成测试.(目前暂不支持 X86 CPU的Linux系统部署)

## 部署准备
### 华为昇腾 NPU 部署环境编译准备
- 1. 软硬件环境满足要求，以及华为昇腾NPU的部署编译环境的准备，请参考：[FastDeploy 华为昇腾NPU部署环境编译准备](../../../../../../docs/cn/build_and_install/huawei_ascend.md.md)  

## 在 华为昇腾NPU 上部署ResNet50_Vd分类模型
请按照以下步骤完成在 华为昇腾NPU 上部署 ResNet50_Vd 模型:
1. 完成[华为昇腾NPU 部署环境编译准备](../../../../../../docs/cn/build_and_install/huawei_ascend.md.md)

2. 运行以下命令完成部署:
```bash
# 下载模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
# 下载图片
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# 运行程序
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg
```

部署成功后输出结果如下：
```bash
ClassifyResult(
label_ids: 153,
scores: 0.685547,
)
#此结果出现后,前后还会出现一些华为昇腾自带的log信息,属正常现象.
```
