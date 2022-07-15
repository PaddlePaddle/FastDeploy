# 编译YOLOv7示例

本文档说明如何进行[YOLOv7](https://github.com/WongKinYiu/yolov7)的快速部署推理。本目录结构如下

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── README.md
│   └── yolov7.cc
├── README.md
└── yolov7.py
```

## 生成ONNX文件

- 手动获取

  访问[YOLOv7](https://github.com/WongKinYiu/yolov7)官方github库，按照指引下载安装，下载`yolov7.pt` 模型，利用 `models/export.py` 得到`onnx`格式文件。

  

  ```
  #下载yolov7模型文件
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
  
  # 导出onnx格式文件
  python models/export.py --grid --dynamic --weights PATH/TO/yolo7.pt
  ```

  

- 从PaddlePaddle获取

## Python部署

### 安装FastDeploy

使用如下命令安装FastDeploy，注意到此处安装的是`vision-cpu`，也可根据需求安装`vision-gpu`

```
# 安装fastdeploy-python工具
pip install fastdeploy-python

# 安装vision-cpu模块
fastdeploy install vision-cpu
```

### 运行demo

```
python yolov7.py
```



## C++部署

### 编译demo文件

```
# 切换到./cpp/ 目录下
cd cpp/

# 下载和解压预测库
wget https://bj.bcebos.com/paddle2onnx/fastdeploy/fastdeploy-linux-x64-0.0.3.tgz
tar xvf fastdeploy-linux-x64-0.0.3.tgz

# 编译示例代码
mkdir build & cd build
cmake ..
make -j
```







