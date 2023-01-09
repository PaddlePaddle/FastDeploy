English | [简体中文](README_CN.md)
# PP-Tracking Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishesshes the deployment of PP-Tracking on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/tracking/pptracking/python

# Download PP-Tracking model files and test videos
wget https://bj.bcebos.com/paddlehub/fastdeploy/fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
tar -xvf fairmot_hrnetv2_w18_dlafpn_30e_576x320.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/person.mp4
# CPU inference
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4 --device cpu
# GPU inference
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4  --device gpu
# TensorRT inference on GPU  （Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model fairmot_hrnetv2_w18_dlafpn_30e_576x320 --video person.mp4  --device gpu --use_trt True
```

## PP-Tracking Python Interface 

```python
fd.vision.tracking.PPTracking(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-Tracking model loading and initialization, among which model_file, params_file, and config_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/deploy/pptracking/cpp/README.md) for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> PPTracking.predict(frame)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **frame**(np.ndarray): Input data in HWC or BGR format. The video frame is obtained through: _,frame=cap.read()

> **Return**
>
> > Return `fastdeploy.vision.MOTResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure

### Class Member Variable
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results



## Other Documents

- [PP-Tracking Model Description](..)
- [PP-Tracking C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
