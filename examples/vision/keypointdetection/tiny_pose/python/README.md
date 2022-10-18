# PP-TinyPose Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`pptinypose_infer.py`快速完成PP-TinyPose在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图单人关键点检测`示例。执行如下脚本即可完成

>> **注意**: PP-Tinypose单模型目前只支持单图单人关键点检测，因此输入的图片应只包含一个人或者进行过裁剪的图像。多人关键点检测请参考[PP-TinyPose Pipeline](../../det_keypoint_unite/python/README.md)

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/keypointdetection/tiny_pose/python

# 下载PP-TinyPose模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/hrnet_demo.jpg

# CPU推理
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device cpu
# GPU推理
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python pptinypose_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --image hrnet_demo.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196386764-dd51ad56-c410-4c54-9580-643f282f5a83.jpeg", width=359px, height=423px />
</div>

## PP-TinyPose Python接口

```python
fd.vision.keypointdetection.PPTinyPose(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-TinyPose模型加载和初始化，其中model_file, params_file以及config_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/EXPORT_MODEL.md)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict函数

> ```python
> PPTinyPose.predict(input_image)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.KeyPointDetectionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性
#### 后处理参数
用户可按照自己的实际需求，修改下列后处理参数，从而影响最终的推理和部署效果

> > * **use_dark**(bool): 是否使用DARK进行后处理[参考论文](https://arxiv.org/abs/1910.06278)


## 其它文档

- [PP-TinyPose 模型介绍](..)
- [PP-TinyPose C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
