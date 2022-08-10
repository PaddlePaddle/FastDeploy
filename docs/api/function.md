# FDTensor C++ 张量化函数

FDTensor是FastDeploy在C++层表示张量的结构体。该结构体主要用于管理推理部署时模型的输入输出数据，支持在不同的Runtime后端中使用。在基于C++的推理部署应用开发过程中，我们往往需要对输入输出的数据进行一些数据处理，用以得到模型的实际输入或者应用的实际输出。这种数据预处理的逻辑可以使用原生的C++标准库来实现，但开发难度会比较大，如对3维Tensor的第2维求最大值。针对这个问题，FastDeploy基于FDTensor开发了一套C++张量化函数，用于降低FastDeploy用户的开发成本，提高开发效率。目前主要分为两类函数：Reduce类函数和Elementwise类函数。

## Reduce类函数

目前FastDeploy支持7种Reduce类函数：Max, Min, Sum, All, Any, Mean, Prod。

### Max

#### 函数签名

```c++
/** Excute the maximum operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Max(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the max value for axis 0 of `inputs`
// The output result would be [[7, 4, 5]].
Max(input, &output, {0}, /* keep_dim = */true);
```

### Min

#### 函数签名

```c++
/** Excute the minimum operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Min(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the min value for axis 0 of `inputs`
// The output result would be [[2, 1, 3]].
Min(input, &output, {0}, /* keep_dim = */true);
```

### Sum

#### 函数签名

```c++
/** Excute the sum operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Sum(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the sum value for axis 0 of `inputs`
// The output result would be [[9, 5, 8]].
Sum(input, &output, {0}, /* keep_dim = */true);
```

### Mean

#### 函数签名

```c++
/** Excute the mean operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Mean(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the mean value for axis 0 of `inputs`
// The output result would be [[4, 2, 4]].
Mean(input, &output, {0}, /* keep_dim = */true);
```

### Prod

#### 函数签名

```c++
/** Excute the product operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Prod(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the product value for axis 0 of `inputs`
// The output result would be [[14, 4, 15]].
Prod(input, &output, {0}, /* keep_dim = */true);
```

### Any

#### 函数签名

```c++
/** Excute the any operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void Any(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::array<bool, 6> bool_inputs = {false, false, true, true, false, true};
input.SetExternalData({2, 3}, FDDataType::INT32, bool_inputs.data());

// Calculate the any value for axis 0 of `inputs`
// The output result would be [[true, false, true]].
Any(input, &output, {0}, /* keep_dim = */true);
```

### All

#### 函数签名

```c++
/** Excute the all operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which will be reduced.
    @param keep_dim Whether to keep the reduced dims, default false.
    @param reduce_all Whether to reduce all dims, default false.
*/
void All(const FDTensor& x, FDTensor* out,
         const std::vector<int64_t>& dims,
         bool keep_dim = false, bool reduce_all = false);
```

#### 使用示例

```c++
FDTensor input, output;
std::array<bool, 6> bool_inputs = {false, false, true, true, false, true};
input.SetExternalData({2, 3}, FDDataType::INT32, bool_inputs.data());

// Calculate the all value for axis 0 of `inputs`
// The output result would be [[false, false, true]].
All(input, &output, {0}, /* keep_dim = */true);
```

## Elementwise类函数

正在开发中，敬请关注······
