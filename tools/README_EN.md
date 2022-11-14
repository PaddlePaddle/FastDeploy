# FastDeploy Toolkit
FastDeploy provides a series of efficient and easy-to-use tools to optimize the deployment experience and improve inference performance.
For example, based on PaddleSlim's Auto Compression Toolkit (ACT), FastDeploy provides users with a one-click model automation compression tool that allows users to easily compress the model with a single command. This document will take FastDeploy's one-click model automation compression tool as an example, introduce how to install the tool, and provide the corresponding documentation for usage.


## FastDeploy One-Click Model Auto Compression Tool

### Environmental Preparation
1.Install PaddlePaddle develop version
```
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
```

2.Install PaddleSlim dev version
```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
python setup.py install
```

3.Install fd-auto-compress package
```bash
# Installing fd-auto-compress via pip
# This tool is included in the python installer of FastDeploy, so you don't need to install it again.
pip install fd-auto-compress==0.0.1

# Execute in the current directory
python setup.py install
```

### The Usage of One-Click Model Auto Compression Tool
After the above steps are successfully installed, you can use FastDeploy one-click model automation compression tool, as shown in the following example.
```bash
fastdeploy --auto_compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model/'
```
For detailed documentation, please refer to [FastDeploy One-Click Model Auto Compression Tool](./auto_compression/README.md)
