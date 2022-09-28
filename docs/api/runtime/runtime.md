# Runtime

在配置`RuntimeOption`后，即可基于不同后端在不同硬件上创建Runtime用于模型推理。

## Python 类

```
class Runtime(runtime_option)
```
**参数**
> * **runtime_option**(fastdeploy.RuntimeOption): 配置好的RuntimeOption类实例

### 成员函数

```
infer(data)
```
根据输入数据进行模型推理

**参数**

> * **data**(dict({str: np.ndarray}): 输入数据，字典dict类型，key为输入名，value为np.ndarray数据类型

**返回值**

返回list, list的长度与原始模型输出个数一致；list中元素为np.ndarray类型


```
num_inputs()
```
返回模型的输入个数

```
num_outputs()
```
返回模型的输出个数


## C++ 类

```
class Runtime
```

### 成员函数

```
bool Init(const RuntimeOption& runtime_option)
```
模型加载初始化

**参数**

> * **runtime_option**: 配置好的RuntimeOption实例

**返回值**

初始化成功返回true，否则返回false


```
bool Infer(vector<FDTensor>& inputs, vector<FDTensor>* outputs)
```
根据输入进行推理，并将结果写回到outputs

**参数**

> * **inputs**: 输入数据
> * **outputs**: 输出数据

**返回值**

推理成功返回true，否则返回false

```
int NumInputs()
```
返回模型输入个数

```
input NumOutputs()
```
返回模型输出个数
