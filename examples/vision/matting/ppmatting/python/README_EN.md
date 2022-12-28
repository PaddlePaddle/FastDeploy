English | [简体中文](README.md)
# PP-Matting Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py`  fast finishes the deployment of PP-Matting on CPU/GPU and GPU accelerated by TensorRT. The script is as follows
```bash
# Download the deployment example code 
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/matting/ppmatting/python

# Download PP-Matting model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-512.tgz
tar -xvf PP-Matting-512.tgz
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
# CPU inference
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device cpu
# GPU inference
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device gpu
# TensorRT inference on GPU（Attention: It is somewhat time-consuming for the operation of model serialization when running TensorRT inference for the first time. Please be patient.）
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device gpu --use_trt True
# kunlunxin XPU inference
python infer.py --model PP-Matting-512 --image matting_input.jpg --bg matting_bgr.jpg --device kunlunxin
```

The visualized result after running is as follows
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>
## PP-Matting Python Interface 

```python
fd.vision.matting.PPMatting(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PP-Matting model loading and initialization, among which model_file, params_file, and config_file are the Paddle inference files exported from the training model. Refer to [Model Export](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/Matting)  for more information

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **config_file**(str): Inference deployment configuration file
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. Paddle format by default

### predict function

> ```python
> PPMatting.predict(input_image)
> ```
>
> Model prediction interface. Input images and output detection results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.MattingResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for the description of the structure.

### Class Member Variable

#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results



## Other Documents

- [PP-Matting Model Description](..)
- [PP-Matting C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/cn/faq/how_to_change_backend.md)
