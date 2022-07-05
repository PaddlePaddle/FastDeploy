# fastdeploy::Runtime

## FDTensor Runtime的输入输出数据结构

```
struct FDTensor {
  std::vector<int64_t> shape; // 形状
  std::string name; // 命名
  FDDataType dtype; // 数据类型
  Device device = Device::CPU; // 数据存放设备

  void* MutableData(); // 获取tensor内存buffer指针

  // 获取tensor数据，如若tensor数据在其它设备
  // 此函数会先将数据拷贝至CPU，再返回指向
  // CPU内存buffer的指针
  void* Data();

  // 初始化Tensor，并复用外部数据指针
  // Tensor的内存buffer将由外部的调用者来创建或释放
  void SetExternalData(const std::vector<int>& new_shape,
                       const FDDataType& data_type,
                       void* data_buffer
                       const Device& dev);

  int Nbytes() const; // 返回tensor数据字节大小

  int Numel() const; // 返回tensor元素个数

  // Debug函数，打印tensor的信息，包含mean、max、min等
  void PrintInfo(const std::string& prefix = "TensorInfo");
};
```

FDTensor是前后处理与`Runtime`进行对接的数据结构，大多情况下建议通过`SetExternalData`来共享用户传入的数据，减小内存拷贝带来的开销。

## Runtime 多后端推理引擎

### RuntimeOption 引擎配置
```
struct RuntimeOption {
  // 模型文件和权重文件
  std::string model_file;
  std::string params_file;
  // 模型格式，当前可支持Frontend::PADDLE / Frontend::ONNX
  Frontend model_format = Frontend::PADDLE;
  Backend backend = Backend::ORT:

  // CPU上运行时的线程数
  int cpu_thread_num = 8;

  // 推理硬件，当前支持Device::CPU / Device::GPU
  // 在CPU/GPU上需与backend进行搭配选择
  Device device;

  // Backend::ORT的参数
  int ort_graph_opt_level;
  int ort_inter_op_num_threads;
  int ort_execution_mode;

  // Backend::TRT的参数
  std::map<std::string, std::vector<int32_t>> trt_fixed_shape;
  std::map<std::string, std::vector<int32_t>> trt_max_shape;
  std::map<std::string, std::vector<int32_t>> trt_min_shape;
  std::map<std::string, std::vector<int32_t>> trt_opt_shape;
  std::string trt_serialize_file = "";
  bool trt_enable_fp16 = false;
  bool trt_enable_int8 = false;
  size_t trt_max_batch_size = 32;
};
```


### Runtime 引擎

```
struct Runtime {
  // 加载模型，引擎初始化
  bool Init(const RuntimeOption& _option);

  // 进行推理
  // 其中输入须正确配置tensor中的name
  bool Infer(std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs);

  int NumInputs(); // 输入个数
  int NumOutputs(); // 输出个数

  TensorInfo GetInputInfo(int index) // 获取输入信息，包括shape, dtype, name
  TensorInfo GetOutputInfo(int index) // 获取输出信息，包括shape, dtype, name

  RuntimeOption option; // 引擎的配置信息
};
```


## Runtime使用示例

### C++

```
#include "fastdeploy/fastdeploy_runtime.h"

int main() {
  auto option = fastdeploy::RuntimeOption();
  option.model_file = "resnet50/inference.pdmodel";
  option.params_file = "resnet50/inference.pdiparams";

  auto runtime = fastdeploy::Runtime();
  assert(runtime.Init(option));

  // 需准备好输入tensor
  std::vector<FDTensor> inputs;

  std::vector<FDTensor> outputs;
  assert(runtime.Infer(tensors, &outputs));

  // 输出tensor的debug信息查看
  outputs[0].PrintInfo();
}
```

### Python

```
import fastdeploy as fd
import numpy as np

option = fd.RuntimeOption();
option.model_file = "resnet50/inference.pdmodel"
option.params_file = "resnet50/inference.pdiparams";

runtime = fd.Runtime(option)

result = runtime.infer({"image": np.random.rand(1, 3, 224, 224)});
```
