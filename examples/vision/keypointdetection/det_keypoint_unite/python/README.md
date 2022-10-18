# PP-PicoDet + PP-TinyPose (Pipeline) Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`det_keypoint_unite_infer.py`快速完成多人模型配置 PP-PicoDet + PP-TinyPose 在CPU/GPU，以及GPU上通过TensorRT加速部署的`单图多人关键点检测`示例。执行如下脚本即可完成
>> **注意**: PP-TinyPose单模型独立部署，请参考[PP-TinyPose 单模型](../../det_keypoint_unite/python/README.md)

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/keypointdetection/det_keypoint_unite/python

# 下载PP-TinyPose模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_TinyPose_256x192_infer.tgz
tar -xvf PP_TinyPose_256x192_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
tar -xvf PP_PicoDet_V2_S_Pedestrian_320x320_infer.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/000000018491.jpg
# CPU推理
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device cpu
# GPU推理
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python det_keypoint_unite_infer.py --tinypose_model_dir PP_TinyPose_256x192_infer --det_model_dir PP_PicoDet_V2_S_Pedestrian_320x320_infer --image 000000018491.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/196393343-eeb6b68f-0bc6-4927-871f-5ac610da7293.jpeg", width=640px, height=427px />
</div>

## PPTinyPosePipeline Python接口

```python
fd.pipeline.PPTinyPose(det_model=None, pptinypose_model=None)
```

PPTinyPosePipeline模型加载和初始化，其中det_model是使用`fd.vision.detection.PicoDet`[参考Detection文档](../../../detection/paddledetection/python/)初始化的检测模型，pptinypose_model是使用`fd.vision.keypointdetection.PPTinyPose`[参考PP-TinyPose文档](../../tiny_pose/python/)初始化的检测模型

**参数**

> * **det_model**(str): 初始化后的检测模型
> * **pptinypose_model**(str): 初始化后的PP-TinyPose模型

### predict函数

> ```python
> PPTinyPosePipeline.predict(input_image)
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

## 其它文档

- [Pipeline 模型介绍](..)
- [Pipeline C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
