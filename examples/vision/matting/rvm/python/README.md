English | [简体中文](README_CN.md)
# RobustVideoMatting Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of RobustVideoMatting on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/matting/rvm/python

# Download RobustVideoMatting model files, test images and videos
## Original ONNX Model
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx
## Specially process the ONNX model for loading TRT
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_trt.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4

# CPU inference
## image
python infer.py --model rvm_mobilenetv3_fp32.onnx --image matting_input.jpg --bg matting_bgr.jpg --device cpu
## video
python infer.py --model rvm_mobilenetv3_fp32.onnx --video video.mp4 --bg matting_bgr.jpg --device cpu
# GPU inference
## image
python infer.py --model rvm_mobilenetv3_fp32.onnx --image matting_input.jpg --bg matting_bgr.jpg --device gpu
## video
python infer.py --model rvm_mobilenetv3_fp32.onnx --video video.mp4 --bg matting_bgr.jpg --device gpu
# TRT inference
## image
python infer.py --model rvm_mobilenetv3_trt.onnx --image matting_input.jpg --bg matting_bgr.jpg --device gpu --use_trt True
## video
python infer.py --model rvm_mobilenetv3_trt.onnx --video video.mp4 --bg matting_bgr.jpg --device gpu --use_trt True
```

The visualized result after running is as follows
<div width="1240">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/19977378/196653716-f7043bd5-dfc2-4e7d-be0f-e12a6af4c55b.gif">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/19977378/196654529-866bff5d-47a2-4584-9627-39b587799228.gif">
</div>

## RobustVideoMatting Python Interface 

```python
fd.vision.matting.RobustVideoMatting(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

RobustVideoMatting model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> RobustVideoMatting.predict(input_image)
> ```
>
> Model prediction interface. Input images and output matting results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.MattingResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.


## Other Documents

- [RobustVideoMatting Model Description](..)
- [RobustVideoMatting C++ Deployment](../cpp)
- [Vision Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
