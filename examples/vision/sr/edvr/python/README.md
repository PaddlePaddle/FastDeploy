English | [简体中文](README_CN.md)
# EDVR Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of EDVR on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/sr/edvr/python

# Download VSR model files and test videos
wget https://bj.bcebos.com/paddlehub/fastdeploy/EDVR_M_wo_tsa_SRx4.tar
tar -xvf EDVR_M_wo_tsa_SRx4.tar
wget https://bj.bcebos.com/paddlehub/fastdeploy/vsr_src.mp4
# CPU inference
python infer.py --model EDVR_M_wo_tsa_SRx4 --video vsr_src.mp4 --frame_num 5 --device cpu
# GPU inference
python infer.py --model EDVR_M_wo_tsa_SRx4 --video vsr_src.mp4 --frame_num 5 --device gpu
# TensorRT inference on GPU (Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.)
python infer.py --model EDVR_M_wo_tsa_SRx4 --video vsr_src.mp4 --frame_num 5 --device gpu --use_trt True
```

## EDVR Python Interface 

```python
fd.vision.sr.EDVR(model_file, params_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

EDVR model loading and initialization, among which model_file and params_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/tutorials/video_super_resolution.md)  for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> EDVR.predict(frames)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **frames**(list[np.ndarray]): Input data in HWC or BGR format. Frames are video frame sequences.

> **Return** list[np.ndarray] is the video frame sequence after SR


## Other Documents

- [EDVR Model Description](..)
- [EDVR C++ Deployment](../cpp)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
