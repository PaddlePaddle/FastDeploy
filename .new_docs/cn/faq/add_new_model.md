# FastDeploy外部模型集成指引

在FastDeploy里面新增一个模型，包括增加C++/Python的部署支持。 本文以torchvision v0.12.0中的ResNet50模型为例，介绍使用FastDeploy做外部[模型集成](#modelsupport)，具体包括如下3步。

| 步骤 | 说明                                | 创建或修改的文件                            |
|:------:|:-------------------------------------:|:---------------------------------------------:|
| [1](#step2)    |  在fastdeploy/vision相应任务模块增加模型实现       | resnet.h、resnet.cc、vision.h                     |
| [2](#step4)     | 通过pybind完成Python接口绑定 | resnet_pybind.cc、classification_pybind.cc |
| [3](#step5)     | 实现Python相应调用接口    | resnet.py、\_\_init\_\_.py                        |

在完成上述3步之后，一个外部模型就集成好了。
<br />
如果您想为FastDeploy贡献代码，还需要为新增模型添加测试代码、说明文档和代码注释，可在[测试](#test)中查看。
## 模型集成     <span id="modelsupport"></span>

### 模型准备  <span id="step1"></span>


在集成外部模型之前，先要将训练好的模型（.pt，.pdparams 等）转换成FastDeploy支持部署的模型格式（.onnx，.pdmodel）。多数开源仓库会提供模型转换脚本，可以直接利用脚本做模型的转换。由于torchvision没有提供转换脚本，因此手动编写转换脚本，本文中将 `torchvison.models.resnet50` 转换为 `resnet50.onnx`， 参考代码如下：

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

### C++部分  <span id="step2"></span>
* 创建`resnet.h`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet.h (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/模型名.h)
  * 创建内容
    * 首先在resnet.h中创建 ResNet类并继承FastDeployModel父类，之后声明`Predict`、`Initialize`、`Preprocess`、`Postprocess`和`构造函数`，以及必要的变量，具体的代码细节请参考[resnet.h](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-69128489e918f305c208476ba793d8167e77de2aa7cadf5dcbac30da448bd28e)。

```C++
class FASTDEPLOY_DECL ResNet : public FastDeployModel {
 public:
  ResNet(...);
  virtual bool Predict(...);
 private:
  bool Initialize();
  bool Preprocess(...);
  bool Postprocess(...);
};
```

* 创建`resnet.cc`文件
  * 创建位置
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet.cc (FastDeploy/C++代码存放位置/视觉模型/任务名称/外部模型/模型名.cc)
  * 创建内容
    * 在`resnet.cc`中实现`resnet.h`中声明函数的具体逻辑，其中`PreProcess` 和 `PostProcess`需要参考源官方库的前后处理逻辑复现，ResNet每个函数具体逻辑如下，具体的代码请参考[resnet.cc](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-d229d702de28345253a53f2a5839fd2c638f3d32fffa6a7d04d23db9da13a871)。

```C++
ResNet::ResNet(...) {
  // 构造函数逻辑
  // 1. 指定 Backend 2. 设置RuntimeOption 3. 调用Initialize()函数
}
bool ResNet::Initialize() {
  // 初始化逻辑
  // 1. 全局变量赋值 2. 调用InitRuntime()函数
  return true;
}
bool ResNet::Preprocess(Mat* mat, FDTensor* output) {
// 前处理逻辑
// 1. Resize 2. BGR2RGB 3. Normalize 4. HWC2CHW 5. 处理结果存入 FDTensor类中  
  return true;
}
bool ResNet::Postprocess(FDTensor& infer_result, ClassifyResult* result, int topk) {
  //后处理逻辑
  // 1. Softmax 2. Choose topk labels 3. 结果存入 ClassifyResult类
  return true;
}
bool ResNet::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
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


### Python部分  <span id="step5"></span>


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

### 编写测试代码
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
您需要在resnet.h文件中为Public函数和变量增加注释，有如下两种注释方式，具体可参考[resnet.h](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-69128489e918f305c208476ba793d8167e77de2aa7cadf5dcbac30da448bd28e)。

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
```
- Python 代码
你需要为resnet.py文件中的函数和变量增加适当的注释，示例如下，具体可参考[resnet.py](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-a4dc5ec2d450e91f1c03819bf314c238b37ac678df56d7dea3aab7feac10a157)。

```python  
# Predict and return the inference result of "input_image".
def predict(self, input_image, topk=1):
    return self._model.predict(input_image, topk)
```

对于集成模型过程中的其他文件，您也可以对实现的细节添加适当的注释说明。
