English | [简体中文](README_CN.md)
# MODNet Python Deployment Example

Before deployment, two steps require confirmation

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of MODNet on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/matting/modnet/python/

# Download modnet model files and test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg

# CPU inference
python infer.py --model modnet_photographic_portrait_matting.onnx --image matting_input.jpg --bg matting_bgr.jpg --device cpu
# GPU inference
python infer.py --model modnet_photographic_portrait_matting.onnx --image matting_input.jpg --bg matting_bgr.jpg --device gpu
# TensorRT inference on GPU 
python infer.py --model modnet_photographic_portrait_matting.onnx --image matting_input.jpg --bg matting_bgr.jpg --device gpu --use_trt True
```

The visualized result after running is as follows

<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186851995-fe9f509f-97d4-4967-a3b0-ce2b3c2f5dca.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186851964-4c9086b9-3490-4fcb-82f9-2106c63aa4f3.jpg">
</div>

## MODNet Python Interface 

```python
fastdeploy.vision.matting.MODNet(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

MODNet model loading and initialization, among which model_file is the exported ONNX model format

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path. No need to set when the model is in ONNX format
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. ONNX format by default

### predict function

> ```python
> MODNet.predict(image_data)
> ```
>
> Model prediction interface. Input images and output matting results.
>
> **Parameter**
>
> > * **image_data**(np.ndarray): Input data in HWC or BGR format

> **Return**
>
> > Return `fastdeploy.vision.MattingResult` structure. Refer to [Vision Model Prediction Results](../../../../../docs/api/vision_results/) for its description.

### Class Member Property
#### Pre-processing Parameter
Users can modify the following pre-processing parameters to their needs, which affects the final inference and deployment results

> > * **size**(list[int]): This parameter changes the size of the resize during preprocessing, containing two integer elements for [width, height] with default value [256, 256]
> > * **alpha**(list[float]): Preprocess normalized alpha, and calculated as `x'=x*alpha+beta`. alpha defaults to [1. / 127.5, 1.f / 127.5, 1. / 127.5]
> > * **beta**(list[float]): Preprocess normalized beta, and calculated as `x'=x*alpha+beta`. beta defaults to [-1.f, -1.f, -1.f]
> > * **swap_rb**(bool): Whether to convert BGR to RGB in pre-processing. Default True



## Other Documents

- [MODNet Model Description](..)
- [MODNet C++ Deployment](../cpp)
- [Model Prediction Results](../../../../../docs/api/vision_results/)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
