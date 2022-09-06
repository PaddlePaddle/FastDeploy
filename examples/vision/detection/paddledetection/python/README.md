# PaddleDetection Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer_xxx.py`快速完成PPYOLOE/PicoDet等模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/paddledetection/python/

#下载PPYOLOE模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# CPU推理
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device cpu
# GPU推理
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device gpu
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示
<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

## PaddleDetection Python接口

```python
fastdeploy.vision.detection.PPYOLOE(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
fastdeploy.vision.detection.PicoDet(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
fastdeploy.vision.detection.PaddleYOLOX(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
fastdeploy.vision.detection.YOLOv3(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
fastdeploy.vision.detection.PPYOLO(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
fastdeploy.vision.detection.FasterRCNN(model_file, params_file, config_file, runtime_option=None, model_format=Frontend.PADDLE)
```

PaddleDetection模型加载和初始化，其中model_file， params_file为导出的Paddle部署模型格式, config_file为PaddleDetection同时导出的部署配置yaml文件

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理配置yaml文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为Paddle

### predict函数

PaddleDetection中各个模型，包括PPYOLOE/PicoDet/PaddleYOLOX/YOLOv3/PPYOLO/FasterRCNN，均提供如下同样的成员函数用于进行图像的检测
> ```python
> PPYOLOE.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
>
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.DetectionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

## 其它文档

- [PaddleDetection 模型介绍](..)
- [PaddleDetection C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
