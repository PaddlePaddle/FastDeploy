
[English](../../en/faq/develop_a_new_model.md) | 中文

# FastDeploy集成新模型流程

在FastDeploy里面新增一个模型，包括增加C++/Python的部署支持。 本文以torchvision v0.12.0中的ResNet50模型为例，介绍使用FastDeploy做外部[模型集成](#modelsupport)，具体包括如下3步。

| 步骤 | 说明                                | 创建或修改的文件                            |
|:------:|:-------------------------------------:|:---------------------------------------------:|
| [1](#step2)    |  在fastdeploy/vision相应任务模块增加模型实现       | yolov7face.h、yolov7face.cc、preprocessor.h、preprocess.cc、postprocessor.h、postprocessor.cc、vision.h                     |
| [2](#step4)     | 通过pybind完成Python接口绑定 | yolov7face_pybind.cc |
| [3](#step5)     | 实现Python相应调用接口    | yolov7face.py、\_\_init\_\_.py                        |

在完成上述3步之后，一个外部模型就集成好了。
<br />
如果您想为FastDeploy贡献代码，还需要为新增模型添加测试代码、说明文档和代码注释，可在[测试](#test)中查看。
## 模型集成     <span id="modelsupport"></span>

### 1 模型准备  <span id="step1"></span>

在集成外部模型之前，先要将训练好的模型（.pt，.pdparams 等）转换成FastDeploy支持部署的模型格式（.onnx，.pdmodel）。多数开源仓库会提供模型转换脚本，可以直接利用脚本做模型的转换。例如yolov7face官方库提供的[export.py](https://github.com/derronqi/yolov7-face/blob/main/models/export.py)文件， 若官方库未提供转换导出文件，则需要手动编写转换脚本，如torchvision没有提供转换脚本，因此手动编写转换脚本，下文中将 `torchvison.models.resnet50` 转换为 `resnet50.onnx`，参考代码如下：

```python
import torch
import torchvision.models as models
model = models.resnet50(pretrained=True)
batch_size = 1  #批处理大小
input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape
model.eval()
x = torch.randn(batch_size, *input_shape)	# 生成张量
export_onnx_file = "resnet50.onnx"			# 目的ONNX文件名
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=12,
                    input_names=["input"],	# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                    "output":{0:"batch_size"}})
```
执行上述脚本将会得到 `resnet50.onnx` 文件。

### 2 CPP代码实现  <span id="step2"></span>
#### 2.1 前处理类实现 
* 创建`preprocessor.h`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/preprocess.h (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/precessor.h)
  * 创建内容
    * 首先在preprocess.h中创建 Yolov7FacePreprocess 类,之后声明`Run`、`preprocess`、`LetterBox`和`构造函数`，以及必要的变量及其`set`和`get`方法，具体的代码细节请参考[preprocess.h](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/preprocessor.h)。

```C++
class FASTDEPLOY_DECL Yolov7FacePreprocessor {
 public:
  Yolov7FacePreprocessor(...);
  bool Run(...);
 protected:
  bool Preprocess(...);
  void LetterBox(...);
};
```

* 创建`preprocessor.cc`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/preprocessor.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/preprocessor.cc)
  * 创建内容
    * 在`preprocessor.cc`中实现`preprocessor.h`中声明函数的具体逻辑，其中`Preprocess`需要参考源官方库的前后处理逻辑复现，preprocessor每个函数具体逻辑如下，具体的代码请参考[preprocessor.cc](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/preprocessor.cc)。

```C++
Yolov7FacePreprocessor::Yolov7FacePreprocessor(...) {
  // 构造函数逻辑
  // 全局变量赋值
}
bool Yolov7FacePreprocessor::Run() {
  // 执行前处理
  // 根据传入图片数量对每张图片处理，并返回结果
  return true;
}
bool Yolov7FacePreprocessor::Preprocess(FDMat* mat, FDTensor* output,
                                        std::map<std::string, std::array<float, 2>>* im_info) {
// 前处理逻辑
// 1. LetterBox 2. convert and permute 3. 处理结果存入 FDTensor类中  
  return true;
}
void Yolov7FacePreprocessor::LetterBox(FDMat* mat) {
  //LetterBox
  return true;
}
```

#### 2.2 后处理类实现
* 创建`postprocessor.h`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/postprocessor.h (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/postprocessor.h)
  * 创建内容
    * 首先在postprocess.h中创建 Yolov7FacePostprocess 类,之后声明`Run`和`构造函数`，以及必要的变量及其`set`和`get`方法，具体的代码细节请参考[postprocessor.h](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/postprocessor.h)。

```C++
class FASTDEPLOY_DECL Yolov7FacePostprocessor {
 public:
  Yolov7FacePostprocessor(...);
  bool Run(...);
};
```

* 创建`postprocessor.cc`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/postprocessor.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/postprocessor.cc)
  * 创建内容
    * 在`postprocessor.cc`中实现`postprocessor.h`中声明函数的具体逻辑，其中`Postprocess`需要参考源官方库的前后处理逻辑复现，postprocessor每个函数具体逻辑如下，具体的代码请参考[postprocessor.cc](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/postprocessor.cc)。

```C++
Yolov7FacePostprocessor::Yolov7FacePostprocessor(...) {
  // 构造函数逻辑
  // 全局变量赋值
}
bool Yolov7FacePostprocessor::Run() {
  // 后处理逻辑
  // 1. Padding 2. Choose box by conf_threshold 3. NMS 4. 结果存入 FaceDetectionResult类
  return true;
}

```
#### 2.3 YOLOv7Face实现
* 创建`yolov7face.h`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/yolov7face.h (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/模型名.h)
  * 创建内容
    * 首先在yolov7face.h中创建 YOLOv7Face 类并继承FastDeployModel父类，之后声明`Predict`、`BatchPredict`、`Initialize`和`构造函数`，以及必要的变量及其`set`和`get`方法，具体的代码细节请参考[yolov7face.h](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/yolov7face.h)。

```C++
class FASTDEPLOY_DECL YOLOv7Face : public FastDeployModel {
 public:
  YOLOv7Face(...);
  virtual bool Predict(...);
  virtual bool BatchPredict(...);
 protected:
  bool Initialize();
  Yolov7FacePreprocessor preprocessor_;
  Yolov7FacePostprocessor postprocessor_;
};
```

* 创建`yolov7face.cc`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/facedet/contrib/yolov7face.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/模型名.cc)
  * 创建内容
    * 在`yolov7face.cc`中实现`yolov7face.h`中声明函数的具体逻辑，YOLOv7Face每个函数具体逻辑如下，具体的代码请参考[yolov7face.cc](https://github.com/CoolKbh/FastDeploy/blob/develop/fastdeploy/vision/facedet/contrib/yolov7face/yolov7face.cc)。

```C++
YOLOv7Face::YOLOv7Face(...) {
  // 构造函数逻辑
  // 1. 指定 Backend 2. 设置RuntimeOption 3. 调用Initialize()函数
}
bool YOLOv7Face::Initialize() {
  // 初始化逻辑
  // 1. 全局变量赋值 2. 调用InitRuntime()函数
  return true;
}
bool ResNet::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  Preprocess(...)
  Infer(...)
  Postprocess(...)
  return true;
}
bool ResNet::BatchPredict(cv::Mat* im, ClassifyResult* result, int topk) {
  Preprocess(...)
  Infer(...)
  Postprocess(...)
  return true;
}
```
<span id="step3"></span>
* 在`vision.h`文件中加入新增模型文件
  * 修改位置
    * FastDeploy/fastdeploy/vision.h
  * 修改内容

```C++
#ifdef ENABLE_VISION
#include "fastdeploy/vision/classification/contrib/resnet.h"
#endif
```


### Pybind部分  <span id="step4"></span>

* 创建Pybind文件  
  * 创建位置
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet_pybind.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/模型名_pybind.cc)
  * 创建内容
    * 利用Pybind将C++中的函数变量绑定到Python中，具体代码请参考[resnet_pybind.cc](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-270af0d65720310e2cfbd5373c391b2110d65c0f4efa547f7b7eeffcb958bdec)。
```C++
void BindResNet(pybind11::module& m) {
  pybind11::class_<vision::classification::ResNet, FastDeployModel>(
      m, "ResNet")
      .def(pybind11::init<std::string, std::string, RuntimeOption, ModelFormat>())
      .def("predict", ...)
      .def_readwrite("size", &vision::classification::ResNet::size)
      .def_readwrite("mean_vals", &vision::classification::ResNet::mean_vals)
      .def_readwrite("std_vals", &vision::classification::ResNet::std_vals);
}
```

* 调用Pybind函数
  * 修改位置
    * FastDeploy/fastdeploy/vision/classification/classification_pybind.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/任务名称}_pybind.cc)
  * 修改内容
```C++
void BindResNet(pybind11::module& m);
void BindClassification(pybind11::module& m) {
  auto classification_module =
      m.def_submodule("classification", "Image classification models.");
  BindResNet(classification_module);
}
```


### Python接口封装  <span id="step5"></span>


* 创建`resnet.py`文件
  * 创建位置
    * FastDeploy/python/fastdeploy/vision/classification/contrib/resnet.py (FastDeploy/Python代码存放位置/fastdeploy/视觉模型/任务名称/外部模型/模型名.py)
  * 创建内容
    * 创建ResNet类继承自FastDeployModel，实现 `\_\_init\_\_`、Pybind绑定的函数（如`predict()`）、以及`对Pybind绑定的全局变量进行赋值和获取的函数`，具体代码请参考[resnet.py](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-a4dc5ec2d450e91f1c03819bf314c238b37ac678df56d7dea3aab7feac10a157)。

```python
class ResNet(FastDeployModel):
    def __init__(self, ...):
        self._model = C.vision.classification.ResNet(...)
    def predict(self, input_image, topk=1):
        return self._model.predict(input_image, topk)
    @property
    def size(self):
        return self._model.size
    @size.setter
    def size(self, wh):
        ...
```
<span id="step6"></span>
* 导入ResNet类
  * 修改位置
    * FastDeploy/python/fastdeploy/vision/classification/\_\_init\_\_.py (FastDeploy/Python代码存放位置/fastdeploy/视觉模型/任务名称/\_\_init\_\_.py)
  * 修改内容

```Python
from .contrib.resnet import ResNet
```

## 测试  <span id="test"></span>
### 编译
  * C++
    * 位置：FastDeploy/

```
mkdir build & cd build
cmake .. -DENABLE_ORT_BACKEND=ON -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=${PWD/fastdeploy-0.0.3
-DENABLE_PADDLE_BACKEND=ON -DENABLE_TRT_BACKEND=ON -DWITH_GPU=ON -DTRT_DIRECTORY=/PATH/TO/TensorRT/
make -j8
make install
```

 编译会得到 build/fastdeploy-0.0.3/。

  * Python
    * 位置：FastDeploy/python/

```
export TRT_DIRECTORY=/PATH/TO/TensorRT/    # 如果用TensorRT 需要填写TensorRT所在位置，并开启 ENABLE_TRT_BACKEND
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_ORT_BACKEND=ON
python setup.py build
python setup.py bdist_wheel
cd dist
pip install fastdeploy_gpu_python-版本号-cpxx-cpxxm-系统架构.whl
```

### 示例代码开发
  * 创建位置: FastDeploy/examples/vision/classification/resnet/ (FastDeploy/示例目录/视觉模型/任务名称/模型名/)
  * 创建目录结构

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── infer.cc    // C++ 版本测试代码
│   └── README.md   // C++版本使用文档
├── python
│   ├── infer.py    // Python 版本测试代码
│   └── README.md   // Python版本使用文档
└── README.md   // ResNet 模型集成说明文档
```

* C++
  * 编写CmakeLists文件、C++ 代码以及 README.md 内容请参考[cpp/](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-afcbe607b796509581f89e38b84190717f1eeda2df0419a2ac9034197ead5f96)。
  * 编译 infer.cc
    * 位置：FastDeploy/examples/vision/classification/resnet/cpp/

```
mkdir build & cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/PATH/TO/FastDeploy/build/fastdeploy-0.0.3/
make
```

* Python
  * Python 代码以及 README.md 内容请参考[python/](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-5a0d6be8c603a8b81454ac14c17fb93555288d9adf92bbe40454449309700135)。

### 为代码添加注释
为了方便用户理解代码，我们需要为新增代码添加注释，添加注释方法可参考如下示例。
- C++ 代码
您需要在resnet.h文件中为函数和变量增加注释，有如下三种注释方式，具体可参考[resnet.h](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-69128489e918f305c208476ba793d8167e77de2aa7cadf5dcbac30da448bd28e)。

```C++
/** \brief Predict for the input "im", the result will be saved in "result".
*
* \param[in] im Input image for inference.
* \param[in] result Saving the inference result.
* \param[in] topk The length of return values, e.g., if topk==2, the result will include the 2 most possible class label for input image.
*/
virtual bool Predict(cv::Mat* im, ClassifyResult* result, int topk = 1);
/// Tuple of (width, height)
std::vector<int> size;
/*! @brief Initialize for ResNet model, assign values to the global variables and call InitRuntime()
*/
bool Initialize();
```
- Python 代码
你需要为resnet.py文件中的函数和变量增加适当的注释，示例如下，具体可参考[resnet.py](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-a4dc5ec2d450e91f1c03819bf314c238b37ac678df56d7dea3aab7feac10a157)。

```python  
  def predict(self, input_image, topk=1):
    """Classify an input image
    :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param topk: (int)The topk result by the classify confidence score, default 1
    :return: ClassifyResult
    """
    return self._model.predict(input_image, topk)
```

对于集成模型过程中的其他文件，您也可以对实现的细节添加适当的注释说明。
