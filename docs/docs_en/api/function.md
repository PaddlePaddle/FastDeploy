# FDTensor C++ Tensor quantization function

FDTensor is FastDeploy's struct that represents the tensor at the C++ level. The struct is mainly used to manage the input and output data of the model during inference deployment and supports different Runtime backends.

In the development of C++-based inference deployment applications, developers often need to process some data on the input and output to get the actual input or the actual output of the application. This pre-processing data logic can easily be done by the original C++ standard library. But it can be difficult to develop, e.g. to find the maximum value for the 2nd dimension of a 3-dimensional Tensor. To solve this problem, FastDeploy has developed a set of C++ tensor functions based on FDTensor to reduce costs and increase efficiency for FastDeploy developers. There are currently four main functions: Reduce, Manipulate, Math and Elementwise.

## Reduce Class Function

Currently FastDeploy supports 7 types of Reduce class functions ：Max, Min, Sum, All, Any, Mean, Prod.

### Max

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the max value for axis 0 of `inputs`
// The output result would be [[7, 4, 5]].
Max(input, &output, {0}, /* keep_dim = */true);
```

### Min

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the min value for axis 0 of `inputs`
// The output result would be [[2, 1, 3]].
Min(input, &output, {0}, /* keep_dim = */true);
```

### Sum

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the sum value for axis 0 of `inputs`
// The output result would be [[9, 5, 8]].
Sum(input, &output, {0}, /* keep_dim = */true);
```

### Mean

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the mean value for axis 0 of `inputs`
// The output result would be [[4, 2, 4]].
Mean(input, &output, {0}, /* keep_dim = */true);
```

### Prod

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::vector<int> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::INT32, inputs.data());

// Calculate the product value for axis 0 of `inputs`
// The output result would be [[14, 4, 15]].
Prod(input, &output, {0}, /* keep_dim = */true);
```

### Any

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::array<bool, 6> bool_inputs = {false, false, true, true, false, true};
input.SetExternalData({2, 3}, FDDataType::INT32, bool_inputs.data());

// Calculate the any value for axis 0 of `inputs`
// The output result would be [[true, false, true]].
Any(input, &output, {0}, /* keep_dim = */true);
```

### All

#### Function Signature

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

#### Demo

```c++
FDTensor input, output;
std::array<bool, 6> bool_inputs = {false, false, true, true, false, true};
input.SetExternalData({2, 3}, FDDataType::INT32, bool_inputs.data());

// Calculate the all value for axis 0 of `inputs`
// The output result would be [[false, false, true]].
All(input, &output, {0}, /* keep_dim = */true);
```

## Manipulate Class Function

Currently FastDeploy supports 1 Manipulate class function: Transpose.

### Transpose

#### Function Signature

```c++
/** Excute the transpose operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param dims The vector of axis which the input tensor will transpose.
*/
void Transpose(const FDTensor& x, FDTensor* out,
               const std::vector<int64_t>& dims);
```

#### Demo

```c++
FDTensor input, output;
std::vector<float> inputs = {2, 4, 3, 7, 1, 5};
input.SetExternalData({2, 3}, FDDataType::FP32, inputs.data());

// Transpose the input tensor with axis {1, 0}.
// The output result would be [[2, 7], [4, 1], [3, 5]]
Transpose(input, &output, {1, 0});
```

## Math Class Function

Currently FastDeploy supports 1 Math class function: Softmax.

### Softmax

#### Function Signature

```c++
/** Excute the softmax operation for input FDTensor along given dims.
    @param x The input tensor.
    @param out The output tensor which stores the result.
    @param axis The axis to be computed softmax value.
*/
void Softmax(const FDTensor& x, FDTensor* out, int axis = -1);
```

#### Demo

```c++
FDTensor input, output;
CheckShape check_shape;
CheckData check_data;
std::vector<float> inputs = {1, 2, 3, 4, 5, 6};
input.SetExternalData({2, 3}, FDDataType::FP32, inputs.data());

// Transpose the input tensor with axis {1, 0}.
// The output result would be
// [[0.04742587, 0.04742587, 0.04742587],
//  [0.95257413, 0.95257413, 0.95257413]]
Softmax(input, &output, 0);
```

## Elementwise Class Function

To be continued...
