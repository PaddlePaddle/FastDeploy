English | [简体中文](README_CN.md)
# AnimeGAN Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

This directory provides examples that `infer.py` fast finishes the deployment of AnimeGAN on CPU/GPU and GPU accelerated by TensorRT. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/generation/anemigan/python
# Download prepared test images
wget https://bj.bcebos.com/paddlehub/fastdeploy/style_transfer_testimg.jpg

# CPU inference
python infer.py --model animegan_v1_hayao_60  --image style_transfer_testimg.jpg  --device cpu
# GPU inference
python infer.py --model animegan_v1_hayao_60 --image style_transfer_testimg.jpg  --device gpu
```

## AnimeGAN Python Interface

```python
fd.vision.generation.AnimeGAN(model_file, params_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

AnimeGAN model loading and initialization, among which model_file and params_file are the model file and parameter file for Paddle inference.

**Parameter**

> * **model_file**(str): Model file path 
> * **params_file**(str): Parameter file path
> * **runtime_option**(RuntimeOption): Backend inference configuration. None by default, which is the default configuration
> * **model_format**(ModelFormat): Model format. PADDLE format by default


### predict function

> ```python
> AnimeGAN.predict(input_image)
> ```
>
> Model prediction interface. Input images and output style transfer results.
>
> **Parameter**
>
> > * **input_image**(np.ndarray): Input data in HWC or BGR format

> **Return** np.ndarray, the image after style transfer in BGR format

### batch_predict function
> ```python
> AnimeGAN.batch_predict function (input_images)
> ```
>
> Model prediction interface. Input a set of images and output style transfer results
>
> **Parameter**
>
> > * **input_images**(list(np.ndarray)): Input data in HWC or BGR format

> **Return** list(np.ndarray), a set of images after style transfer in BGR format

## Other Documents

- [Style Transfer Model Description](..)
- [C++ Deployment](../cpp)
- [How to switch the model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
