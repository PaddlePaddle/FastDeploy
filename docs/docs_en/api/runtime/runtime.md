# Runtime

After configuring `RuntimeOption`, developers can create Runtime for model inference on different hardware based on different backends.

## Python Class

```
class Runtime(runtime_option)
```

**Parameters**

> * **runtime_option**(fastdeploy.RuntimeOption): Configured RuntimeOption class and instance.

### Member function

```
infer(data)
```

Model inference based on input data

**Parameters**

> * **data**(dict({str: np.ndarray}): Input dict data, and key is input name, value is np.ndarray

**Return Value**

Returns a list, whose length equals the number of elements in the original model; elements in the list are np.ndarray

```
num_inputs()
```

Input number that returns to the model

```
num_outputs()
```

Output number that returns to the model

## C++  Class

```
class Runtime
```

### Member function

```
bool Init(const RuntimeOption& runtime_option)
```

Model loading initialization

**Parameters**

> * **runtime_option**: Configured RuntimeOption class and instance

**Return Value**

Returns TRUE for successful initialisation, FALSE otherwise

```
bool Infer(vector<FDTensor>& inputs, vector<FDTensor>* outputs)
```

Inference from the input and write the result to outputs

**Parameters**

> * **inputs**: Input data
> * **outputs**: Output data

**Return Value**

Returns TRUE for successful inference, FALSE otherwise

```
int NumInputs()
```

Input number that returns to the model

```
input NumOutputs()
```

Output number that returns to the model
