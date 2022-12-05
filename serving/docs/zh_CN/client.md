# 客户端访问说明
本文以访问使用fastdeployserver部署的yolov5模型为例，讲述客户端如何请求服务端进行推理服务。关于如何使用fastdeployserver部署yolov5模型，可以参考文档[yolov5服务化部署](../../../examples/vision/detection/yolov5/serving)

## 基本原理介绍
fastdeployserver实现了由[kserve](https://github.com/kserve/kserve)提出的为机器学习模型推理服务而设计的[Predict Protocol协议](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) API，该API既简单易用同时又支持高性能部署的使用场景，目前提供基于HTTP和GRPC两种网络协议的访问方式。这里主要介绍如何使用HTTP进行访问。

当fastdeployserver启动后，默认情况下，8000端口用于响应HTTP请求，8001端口用于响应GRPC请求。用户需要请求的资源通常有两种：

1. **模型的元信息（metadata)**
访问方式： GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`

使用GET请求该url路径可以获取参与服务的模型的元信息，其中`${MODEL_NAME}`表示模型的名字，${MODEL_VERSION}表示模型的版本。服务器会把模型的元信息以json格式返回，返回的格式为一个字典，以$metadata_model_response表示返回的对象，各字段和内容形式表示如下：

```json
$metadata_model_response =
    {
      "name" : $string,
      "versions" : [ $string, ... ] #optional,
      "platform" : $string,
      "inputs" : [ $metadata_tensor, ... ],
      "outputs" : [ $metadata_tensor, ... ]
    }

$metadata_tensor =
    {
      "name" : $string,
      "datatype" : $string,
      "shape" : [ $number, ... ]
    }
```

2. **推理服务**

访问方式：POST `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer`

使用POST请求该url路径可以请求模型的推理服务，获取推理结果。POST请求中的数据同样以json格式上传，以$inference_request表示上传的对象，各字段和内容形式表示如下：
```json
 $inference_request =
    {
      "id" : $string #optional,
      "parameters" : $parameters #optional,
      "inputs" : [ $request_input, ... ],
      "outputs" : [ $request_output, ... ] #optional
    }

$request_input =
    {
      "name" : $string,
      "shape" : [ $number, ... ],
      "datatype"  : $string,
      "parameters" : $parameters #optional,
      "data" : $tensor_data
    }

$request_output =
    {
      "name" : $string,
      "parameters" : $parameters #optional,
    }

$parameters =
{
  $parameter, ...
}

$parameter = $string : $string | $number | $boolean
```
其中$tensor_data表示一维或多维数组，如果是一维数据，必须按照行主序的方式进行排列tensor中的数据。
服务器推理完成后，返回结果数据，以$inference_response表示返回的对象，各字段和内容形式表示如下：

```json
$inference_response =
    {
      "model_name" : $string,
      "model_version" : $string #optional,
      "id" : $string,
      "parameters" : $parameters #optional,
      "outputs" : [ $response_output, ... ]
    }

$response_output =
    {
      "name" : $string,
      "shape" : [ $number, ... ],
      "datatype"  : $string,
      "parameters" : $parameters #optional,
      "data" : $tensor_data
    }
```

## 客户端工具
了解了fastdeployserver服务提供的接口之后，用户可以用任何工具使用HTTP来请求服务器。这里分别介绍如何使用tritonclient和requests库来访问fastdeployserver服务，第一种工具是专门为模型服务做的客户端，对请求和响应进行了封装，方便用户使用。而第二种工具通用的http客户端工具，使用该工具进行访问可以帮助用户更好地理解上述过程。

1. 使用tritonclient访问服务
安装tritonclient\[http\]
```bash
pip install tritonclient[http]
```

- 1. 获取yolov5的模型元数据
```python
import tritonclient.http as httpclient  # 导入httpclient
server_addr = 'localhost:8000'  # 这里写fastdeployserver服务器的实际地址
client = httpclient.InferenceServerClient(server_addr)  # 创建client
model_metadata = client.get_model_metadata(
      model_name='yolov5', model_version='1') # 请求yolov5模型的元数据
```
- 2. 请求推理服务
根据返回的model_metadata来构造请求数据。首先看一下模型的输入和输出有哪些
```python
print(model_metadata.inputs)
```
```text
[{'name': 'INPUT', 'datatype': 'UINT8', 'shape': [-1, -1, -1, 3]}]
```python
print(model_metadata.outputs)
```text
[{'name': 'detction_result', 'datatype': 'BYTES', 'shape': [-1, -1]}]
```
根据模型的inputs和outputs构造数据，然后请求推理
```
# 假设图像数据的文件名为000000014439.jpg
import cv2
image = cv2.imread('000000014439.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]

inputs = []
infer_input = httpclient.InferInput('INPUT', image.shape, 'UINT8') # 构造输入
infer_input.set_data_from_numpy(image)  # 载入输入数据
inputs.append(infer_input)
outputs = []
infer_output = httpclient.InferRequestedOutput('detction_result') # 构造输出
outputs.append(infer_output)
response = client.infer(
            'yolov5', inputs, model_version='1', outputs=outputs)  # 请求推理
response_outputs = response.as_numpy('detction_result') # 根据输出变量名获取结果
```

2. 使用requests访问服务
安装requests
```bash
pip install requests
```
- 1. 获取yolov5的模型元数据
```python
import requests
url = 'http://localhost:8000/v2/models/yolov5/versions/1' # 根据上述章节中"模型的元信息"的获取接口构造url
response = requests.get(url)
response = response.json() # 返回数据为json，以json格式解析
```
打印一下返回的模型元数据
```python
print(response)
```
```text
{'name': 'yolov5', 'versions': ['1'], 'platform': 'ensemble', 'inputs': [{'name': 'INPUT', 'datatype': 'UINT8', 'shape': [-1, -1, -1, 3]}], 'outputs': [{'name': 'detction_result', 'datatype': 'BYTES', 'shape': [-1, -1]}]}
```
- 2. 请求推理服务
根据模型的inputs和outputs构造数据，然后请求推理。
```python
url = 'http://localhost:8000/v2/models/yolov5/versions/1/infer' # 根据上述章节中"推理服务"的接口构造url
# 假设图像数据的文件名为000000014439.jpg
import cv2
image = cv2.imread('000000014439.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]

payload = {
  "inputs" : [
    {
      "name" : "INPUT",
      "shape" : image.shape,
      "datatype" : "UINT8",
      "data" : image.tolist()
    }
  ],
  "outputs" : [
    {
      "name" : "detction_result"
    }
  ]
}
response = requests.post(url, data=json.dumps(payload))
response = response.json()  # 返回数据为json，以json格式解析后即为推理后返回的结果
```
