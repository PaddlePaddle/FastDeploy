English | [简体中文](README_CN.md)
# Petr Python Deployment Example

Before deployment, the following two steps need to be confirmed

- 1. The hardware and software environment meets the requirements, refer to [FastDeploy environment requirements](../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)
- 2. FastDeploy Python whl package installation, refer to [FastDeploy Python Installation](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

This directory provides an example of `infer.py` to quickly complete the deployment of Petr on CPU/GPU. Execute the following script to complete

```bash
#Download deployment sample code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/vision/paddle3d/petr/python

wget https://bj.bcebos.com/fastdeploy/models/petr.tar.gz
tar -xf petr.tar.gz
wget https://bj.bcebos.com/fastdeploy/models/petr_test.png

# CPU reasoning
python infer.py --model petr --image petr_test.png --device cpu
# GPU inference
python infer.py --model petr --image petr_test.png --device gpu
```

## Petr Python interface

```python
fastdeploy.vision.detection.Petr(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

Petr model loading and initialization.

**parameter**
> * **model_file**(str): model file path
> * **params_file**(str): parameter file path
> * **config_file**(str): configuration file path
> * **runtime_option**(RuntimeOption): Backend reasoning configuration, the default is None, that is, the default configuration is used
> * **model_format**(ModelFormat): model format, the default is Paddle format

### predict function

> ```python
> Petr. predict(image_data)
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

- [Petr Model Introduction](..)
- [Petr C++ deployment](../cpp)
- [Description of model prediction results](../../../../../docs/api/vision_results/)
- [How to switch model inference backend engine](../../../../../docs/en/faq/how_to_change_backend.md)
