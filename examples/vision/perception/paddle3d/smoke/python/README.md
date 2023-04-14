English | [简体中文](README_CN.md)
# Smoke Python Deployment Example

Before deployment, the following two steps need to be confirmed

- 1. The hardware and software environment meets the requirements, refer to [FastDeploy environment requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. FastDeploy Python whl package installation, refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides an example of `infer.py` to quickly complete the deployment of Smoke on CPU/GPU. Execute the following script to complete

```bash
#Download deployment sample code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/vision/paddle3d/smoke/python

wget https://bj.bcebos.com/fastdeploy/models/smoke.tar.gz
tar -xf smoke.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/smoke_test.png

# CPU reasoning
python infer.py --model smoke --image smoke_test.png --device cpu
# GPU inference
python infer.py --model smoke --image smoke_test.png --device gpu
```

The visual result after running is shown in the figure below

<img width="640" src="https://user-images.githubusercontent.com/30516196/230387825-53ac0a09-4137-4e49-9564-197cbc30ff08.png">

## Smoke Python interface

```python
fastdeploy.vision.detection.Smoke(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

Smoke model loading and initialization.

**parameter**
> * **model_file**(str): model file path
> * **params_file**(str): parameter file path
> * **config_file**(str): configuration file path
> * **runtime_option**(RuntimeOption): Backend reasoning configuration, the default is None, that is, the default configuration is used
> * **model_format**(ModelFormat): model format, the default is Paddle format

### predict function

> ```python
> Smoke. predict(image_data)
> ```
>
> Model prediction interface, the input image directly outputs the detection result.
>
> **parameters**
>
> > * **image_data**(np.ndarray): input data, note that it must be in HWC, BGR format

> **Back**
>
> > Return the `fastdeploy.vision.PerceptionResult` structure, structure description reference document [Vision Model Prediction Results](../../../../../docs/api/vision_results/)


## Other documents

- [Smoke Model Introduction](..)
- [Smoke C++ deployment](../cpp)
- [Description of model prediction results](../../../../../docs/api/vision_results/)
- [How to switch model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
