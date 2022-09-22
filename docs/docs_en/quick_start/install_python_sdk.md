# FastDeploy Python SDK

FastDeploy provides pre-compiled Python Wheel packages on Windows/Linux/Mac. Developers can download and install them directly, or compile the code manually.

Currently, Fast Deploy supports

- Python3.6-3.9 on Linux
- Python3.8-3.9 on Windows
- Python3.6-3.9 on Mac

## Install CPU Python

```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## Install GPU Python

```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## Notes:

- Please choose either `fastdeploy-python`or `fastdeploy-gpu-python`for installation.
- If you have installed CPU `fastdeploy-python`, please execute `pip uninstall fastdeploy-python` to uninstall the existing version to install GPU `fastdeploy-gpu-python`. 

## Dependencies

- cuda >= 11.2
- cudnn >= 8.0

## Other related docs

- [FastDeploy Prebuilt C++ Libraries](./install_cpp_sdk.md)
- [Example Vision and NLP Model deployment with C++/Python](../../../examples/vision/)
