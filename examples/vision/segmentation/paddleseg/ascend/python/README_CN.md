[English](README.md) | 简体中文
# PaddleSeg Python部署示例

本目录下提供`infer.py`快速完成PP-LiteSeg在华为昇腾上部署的示例。

在部署前，需自行编译基于华为昇腾NPU的FastDeploy python wheel包，参考文档[华为昇腾NPU部署环境编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/huawei_ascend.md)，编译python wheel包并安装

>>**注意** **PP-Matting**、**PP-HumanMatting**的模型，请从[Matting模型部署](../../../matting)下载


```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/ascend/cpp

# 下载PP-LiteSeg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
tar -xvf PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# 华为昇腾推理
python infer.py --model PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image cityscapes_demo.png
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

## PaddleSegModel Python接口

```python
fd.vision.segmentation.PaddleSegModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleSeg模型加载和初始化，其中model_file, params_file以及config_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/docs/model_export_cn.md)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict函数

> ```python
> PaddleSegModel.predict(input_image)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.SegmentationResult`结构体，SegmentationResult结构体说明参考[SegmentationResult结构体介绍](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/segmentation_result_CN.md)

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **is_vertical_screen**(bool): PP-HumanSeg系列模型通过设置此参数为`true`表明输入图片是竖屏，即height大于width的图片

#### 后处理参数
> > * **apply_softmax**(bool): 当模型导出时，并未指定`apply_softmax`参数，可通过此设置此参数为`true`，将预测的输出分割标签（label_map）对应的概率结果(score_map)做softmax归一化处理

## 快速链接

- [PaddleSeg 模型介绍](..)
- [PaddleSeg C++部署](../cpp)

## 常见问题
- [如何将模型预测结果SegmentationResult转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/segmentation_result_CN.md)
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [PaddleSeg python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/semantic_segmentation.html)
