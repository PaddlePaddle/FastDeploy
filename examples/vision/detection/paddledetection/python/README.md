English | [简体中文](README_CN.md)
# PaddleDetection Python Deployment Example

Before deployment, two steps require confirmation.

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer_xxx.py` fast finishes the deployment of PPYOLOE/PicoDet models on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/python/

# Download the PPYOLOE model file and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
tar xvf ppyoloe_crn_l_300e_coco.tgz

# CPU inference
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device cpu
# GPU inference
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device gpu
# TensorRT inference on GPU  （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device gpu --use_trt True
# Kunlunxin XPU Inference
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device kunlunxin
# Huawei Ascend Inference
python infer_ppyoloe.py --model_dir ppyoloe_crn_l_300e_coco --image 000000014439.jpg --device ascend
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/19339784/184326520-7075e907-10ed-4fad-93f8-52d0e35d4964.jpg", width=480px, height=320px />
</div>

## PaddleDetection Python Interface

```python
fastdeploy.vision.detection.PPYOLOE(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PicoDet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOX(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.YOLOv3(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PPYOLO(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.FasterRCNN(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.MaskRCNN(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.SSD(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv5(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv6(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.PaddleYOLOv7(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
fastdeploy.vision.detection.RTMDet(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleDetection model loading and initialization, among which model_file and params_file are the exported Paddle model format. config_file is the configuration yaml file exported by PaddleDetection simultaneously

**Parameter**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference configuration yaml file path
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default. (use the default configuration)
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict Function

PaddleDetection models, including PPYOLOE/PicoDet/PaddleYOLOX/YOLOv3/PPYOLO/FasterRCNN, all provide the following member functions for image detection
> ```python
> PPYOLOE.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
>
> Model prediction interface. Input images and output results directly.
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.DetectionResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.

## Other Documents

- [PaddleDetection Model Description](..)
- [PaddleDetection C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
