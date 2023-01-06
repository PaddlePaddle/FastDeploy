English | [简体中文](README_CN.md)
# PaddleSeg Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

【Attention】For the deployment of  **PP-Matting**、**PP-HumanMatting** and **ModNet**, refer to [Matting Model Deployment](../../../matting)

This directory provides examples that `infer.py`  fast finishes the deployment of Unet on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
```bash
# Download the deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

# Download Unet model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/Unet_cityscapes_without_argmax_infer.tgz
tar -xvf Unet_cityscapes_without_argmax_infer.tgz
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# CPU inference
python infer.py --model Unet_cityscapes_without_argmax_infer --image cityscapes_demo.png --device cpu
# GPU inference
python infer.py --model Unet_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu
# TensorRT inference on GPU（Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model Unet_cityscapes_without_argmax_infer --image cityscapes_demo.png --device gpu --use_trt True
# kunlunxin XPU inference
python infer.py --model Unet_cityscapes_without_argmax_infer --image cityscapes_demo.png --device kunlunxin
```

The visualized result after running is as follows
<div  align="center">  
<img src="https://user-images.githubusercontent.com/16222477/191712880-91ae128d-247a-43e0-b1e3-cafae78431e0.jpg", width=512px, height=256px />
</div>

## PaddleSegModel Python Interface 

```python
fd.vision.segmentation.PaddleSegModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleSeg model loading and initialization, among which model_file, params_file, and config_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/docs/model_export_cn.md)  for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> PaddleSegModel.predict(input_image)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.SegmentationResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **is_vertical_screen**(bool): For PP-HumanSeg models, the input image is portrait with height greater than width by setting this parameter to `true`
#### Post-processing Parameter
> > * **apply_softmax**(bool): The `apply_softmax` parameter is not specified when the model is exported. Set this parameter to `true` to normalize the probability result (score_map) of the predicted output segmentation label (label_map) in softmax

## Other Documents

- [PaddleSeg Model Description](..)
- [PaddleSeg C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
