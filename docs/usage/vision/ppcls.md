# PaddleClas分类模型推理

PaddleClas模型导出参考[PaddleClas](https://github.com/PaddlePaddle/PaddleClas.git)

## Python API说明

### Model类
```
fastdeploy.vision.ppcls.Model(model_file, params_file, config_file, runtime_option=None, model_format=fastdeploy.Frontend.PADDLE)
```

**参数**

> * **model_file**(str): 模型文件，如resnet50/inference.pdmodel  
> * **params_file**(str): 参数文件，如resnet50/inference.pdiparams  
> * **config_file**(str): 配置文件，来源于PaddleClas提供的推理配置文件，如[inference_cls.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/configs/inference_cls.yaml)  
> * **runtime_option**(fd.RuntimeOption): 后端推理的配置, 默认为None，即采用默认配置  
> * **model_format**(fd.Frontend): 模型格式说明，PaddleClas的模型格式均为Frontend.PADDLE  

#### predict接口
```
Model.predict(image_data, topk=1)
```

> **参数**
>
> > * **image_data**(np.ndarray): 输入数据, 注意需为HWC，RGB格式  
> > * **topk**(int): 取前top的分类  

> **返回结果**
>
> > * **result**(ClassifyResult)：结构体包含`label_ids`和`scores`两个list成员变量，表示类别，和各类别对应的置信度

### 示例

> ```
> import fastdeploy.vision as vis
> import cv2
> model = vis.ppcls.Model("resnet50/inference.pdmodel", "resnet50/inference.pdiparams", "resnet50/inference_cls.yaml")
> im = cv2.imread("test.jpeg")
> result = model.predict(im, topk=5)
> print(result.label_ids[0], result.scores[0])
> ```

## C++ API说明

需添加头文件`#include "fastdeploy/vision.h"`

### Model类

```
fastdeploy::vision::ppcls::Model(
                    const std::string& model_file,
                    const std::string& params_file,
                    const std::string& config_file,
                    const RuntimeOption& custom_option = RuntimeOption(),
                    const Frontend& model_format = Frontend::PADDLE)
```

**参数**
> * **model_file**: 模型文件，如resnet50/inference.pdmodel  
> * **params_file**: 参数文件，如resnet50/inference.pdiparams  
> * **config_file**: 配置文件，来源于PaddleClas提供的推理配置文件，如[inference_cls.yaml](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/configs/inference_cls.yaml)  
> * **runtime_option**: 后端推理的配置, 不设置的情况下，采用默认配置  
> * **model_format**: 模型格式说明，PaddleClas的模型格式均为Frontend.PADDLE  

#### Predict接口
```
bool Model::Predict(cv::Mat* im, ClassifyResult* result, int topk = 1)
```

> **参数**
> > * **im**: 输入图像数据，须为HWC，RGB格式(注意传入的im在预处理过程中会被修改)  
> > * **result**: 分类结果  
> > * **topk**: 取分类结果前topk  

> **返回结果**
> > true或false，表示预测成功与否

### 示例
> ```
> #include "fastdeploy/vision.h"
>
> int main() {
>   typedef vis = fastdeploy::vision;
>   auto model = vis::ppcls::Model("resnet50/inference.pdmodel", "resnet50/inference.pdiparams", "resnet50/inference_cls.yaml");
>
>   if (!model.Initialized()) {
>     std::cerr << "Initialize failed." << std::endl;
>     return -1;
>   }
>
>   cv::Mat im = cv::imread("test.jpeg");
>
>   vis::ClassifyResult res;
>   if (!model.Predict(&im, &res, 5)) {
>     std::cerr << "Prediction failed." << std::endl;
>     return -1;
>   }
>
>   std::cout << res.label_ids[0] << " " << res.scores[0] << std::endl;
>   return 0;
> }
```
