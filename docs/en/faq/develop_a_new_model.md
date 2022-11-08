# How to Integrate New Model on FastDeploy

 How to add a new model on FastDeploy, including C++/Python deployment?  Here, we take the ResNet50 model in torchvision v0.12.0 as an example, introducing external [Model Integration](#modelsupport) on FastDeploy. The whole process only needs 3 steps.

| Step        | Description                                                                      | Create or modify the files                |
|:-----------:|:--------------------------------------------------------------------------------:|:-----------------------------------------:|
| [1](#step2) | Add a model implementation to the corresponding task module in FastDeploy/vision | resnet.h、resnet.cc、vision.h               |
| [2](#step4) | Python interface binding via pybind                                              | resnet_pybind.cc、classification_pybind.cc |
| [3](#step5) | Use Python to call Interface                                                     | resnet.py、\_\_init\_\_.py                 |

After completing the above 3 steps, an external model is integrated.

If you want to contribute your code to FastDeploy, it is very kind of you to add test code, instructions (Readme), and code annotations for the added model in the [test](#test).

## Model Integration

### Prepare the models

Before integrating external models, it is important to convert the trained models (.pt, .pdparams, etc.) to the model formats (.onnx, .pdmodel) that FastDeploy supports for deployment. Most open source repositories provide model conversion scripts for developers. As torchvision does not provide conversion scripts, developers can write conversion scripts manually. In this demo, we convert `torchvison.models.resnet50` to `resnet50.onnx` with the following code for your reference.

```python
import torch
import torchvision.models as models
model = models.resnet50(pretrained=True)
batch_size = 1  #batch size
input_shape = (3, 224, 224)   #Input data, change to your own input shape
model.eval()
x = torch.randn(batch_size, *input_shape)    # Generate Tensor
export_onnx_file = "resnet50.onnx"            # ONNX file name
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=12,
                    input_names=["input"],    # Input names
                    output_names=["output"],    # Output names
                    dynamic_axes={"input":{0:"batch_size"},  # batch size variables
                                    "output":{0:"batch_size"}})
```

Running the above script will generate a`resnet50.onnx` file.

### C++

* Create`resnet.h` file
  * Create a path
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet.h (FastDeploy/C++ code/vision/task name/external model name/model name.h)
  * Create content
    * First, create ResNet class in resnet.h and inherit from FastDeployModel parent class, then declare `Predict`, `Initialize`, `Preprocess`, `Postprocess` and `Constructor`, and necessary variables, please refer to [resnet.h](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-69128489e918f305c208476ba793d8167e77de2aa7cadf5dcbac30da448bd28e)  for details.

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

* Create`resnet.cc` file
  * Create a path
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet.cc (FastDeploy/C++ code/vision/task name/external model name/model name.cc)
  * Create content
    * Implement the specific logic of the functions declared in `resnet.h` to `resnet.cc`, where `PreProcess` and `PostProcess` need to refer to the official source library for pre- and post-processing logic reproduction. The specific logic of each ResNet function is as follows. For more detailed code, please refer to [resnet.cc](https:// github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-d229d702de28345253a53f2a5839fd2c638f3d32fffa6a7d04d23db9da13a871).

```C++
ResNet::ResNet(...) {
  // Constructor logic
  // 1. Specify Backend 2. Set RuntimeOption 3. Call Initialize()function
}
bool ResNet::Initialize() {
  // Initialization logic
  // 1. Assign values to global variables 2. Call InitRuntime()function
  return true;
}
bool ResNet::Preprocess(Mat* mat, FDTensor* output) {
// Preprocess logic
// 1. Resize 2. BGR2RGB 3. Normalize 4. HWC2CHW 5. save the results to FDTensor class  
  return true;
}
bool ResNet::Postprocess(FDTensor& infer_result, ClassifyResult* result, int topk) {
  //Postprocess logic
  // 1. Softmax 2. Choose topk labels 3. Save the results to ClassifyResult
  return true;
}
bool ResNet::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  Preprocess(...)
  Infer(...)
  Postprocess(...)
  return true;
}
```

* Add new model file to`vision.h`
  * modify location
    * FastDeploy/fastdeploy/vision.h
  * modify content

```C++
#ifdef ENABLE_VISION
#include "fastdeploy/vision/classification/contrib/resnet.h"
#endif
```

### Pybind

* Create Pybind file
  
  * Create path
    
    * FastDeploy/fastdeploy/vision/classification/contrib/resnet_pybind.cc (FastDeploy/C++ code/vision model/taks name/external model/model name_pybind.cc)
  
  * Create content
    
    * Use Pybind to bind function variables from C++ to Python, please refer to [resnet_pybind.cc](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-270af0d65720310e2cfbd5373c391b2110d65c0f4efa547f7b7eeffcb958bdec) for more details.
      
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

* Call Pybind function
  
  * modify path
    
    * FastDeploy/fastdeploy/vision/classification/classification_pybind.cc (FastDeploy/C++ code/vision model/task name/task name}_pybind.cc)
  
  * modify content
    
    ```C++
    void BindResNet(pybind11::module& m);
    void BindClassification(pybind11::module& m) {
    auto classification_module =
      m.def_submodule("classification", "Image classification models.");
    BindResNet(classification_module);
    }
    ```

### Python

* Create`resnet.py`file
  * Create path
    * FastDeploy/python/fastdeploy/vision/classification/contrib/resnet.py (FastDeploy/Python code/fastdeploy/vision model/task name/external model/model name.py)
  * Create content
    * Create ResNet class inherited from FastDeployModel, and implement `\_\_init\_\_`, Pybind bonded functions (such as `predict()`), and `functions to assign and get global variables bound to Pybind`. Please refer to [resnet.py](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-a4dc5ec2d450e91f1c03819bf314c238b37ac678df56d7dea3aab7feac10a157) for details 

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

* Import ResNet classes
  * modify path
    * FastDeploy/python/fastdeploy/vision/classification/\_\_init\_\_.py (FastDeploy/Python code/fastdeploy/vision model/task name/\_\_init\_\_.py)
  * modify content

```Python
from .contrib.resnet import ResNet
```

## Test

### Compile

* C++
  * Path：FastDeploy/

```
mkdir build & cd build
cmake .. -DENABLE_ORT_BACKEND=ON -DENABLE_VISION=ON -DCMAKE_INSTALL_PREFIX=${PWD/fastdeploy-0.0.3
-DENABLE_PADDLE_BACKEND=ON -DENABLE_TRT_BACKEND=ON -DWITH_GPU=ON -DTRT_DIRECTORY=/PATH/TO/TensorRT/
make -j8
make install
```

 Compile to get build/fastdeploy-0.0.3/。

* Python
  * Path：FastDeploy/python/

```
export TRT_DIRECTORY=/PATH/TO/TensorRT/    #If TensorRT is used, developers need to fill in the location of TensorRT and enable ENABLE_TRT_BACKEND
export ENABLE_TRT_BACKEND=ON
export WITH_GPU=ON
export ENABLE_PADDLE_BACKEND=ON
export ENABLE_OPENVINO_BACKEND=ON
export ENABLE_VISION=ON
export ENABLE_ORT_BACKEND=ON
python setup.py build
python setup.py bdist_wheel
cd dist
pip install fastdeploy_gpu_python-Version number-cpxx-cpxxm-system architecture.whl
```

### Compile Test Code

* Create path: FastDeploy/examples/vision/classification/resnet/ (FastDeploy/examples/vision model/task anme/model name/)
* Creating directory structure

```
.
├── cpp
│   ├── CMakeLists.txt
│   ├── infer.cc    // C++ test code
│   └── README.md   // C++ Readme
├── python
│   ├── infer.py    // Python test code
│   └── README.md   // Python Readme
└── README.md   // ResNet model integration readme
```

* C++
  * Write CmakeLists、C++  code and README.md . Please refer to[cpp/](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-afcbe607b796509581f89e38b84190717f1eeda2df0419a2ac9034197ead5f96)
  * Compile infer.cc
    * Path：FastDeploy/examples/vision/classification/resnet/cpp/

```
mkdir build & cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=/PATH/TO/FastDeploy/build/fastdeploy-0.0.3/
make
```

* Python
  * Please refer to[python/](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-5a0d6be8c603a8b81454ac14c17fb93555288d9adf92bbe40454449309700135) for Python code and Readme.md

### Annotate the Code



To make the code clear for understanding,  developers can annotate the newly-added code. 

- C++ code
  Developers need to add annotations for functions and variables in the resnet.h file, there are three annotating methods as follows, please refer to [resnet.h](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff- 69128489e918f305c208476ba793d8167e77de2aa7cadf5dcbac30da448bd28e) for more details.

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

- Python 
  The following example is to demonstrate how to annotate functions and variables in resnet.py file. For more details, please refer to [resnet.py](https://github.com/PaddlePaddle/FastDeploy/pull/347/files#diff-a4dc5ec2d450e91f1c03819bf314c238b37ac678df56d7dea3aab7feac10a157). 

```python
  def predict(self, input_image, topk=1):
    """Classify an input image
    :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
    :param topk: (int)The topk result by the classify confidence score, default 1
    :return: ClassifyResult
    """
    return self._model.predict(input_image, topk)
```

Other files in the integration process can also be annotated to explain the details of the implementation.
