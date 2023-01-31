English | [中文](../zh_CN/client.md)

# Client Access Instruction
Let us take accessing a YOLOv5 model deployed by fastdeployserver as an example, and describe how the client requests the server for inference services. For how to deploy a YOLOv5 model using fastdeployserver, you can refer to [YOLOv5 service-based deployment](../../../examples/vision/detection/yolov5/serving).

## Fundamental Introduction
Fastdeployserver implements the [Predict Protocol](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) proposed by [kserve](https://github.com/kserve/kserve) , which is an API designed for machine learning model inference service. It is easy to use while supporting high performance deployment scenarios. Currently the API provides access based on both HTTP and GRPC.


When fastdeployserver starts, port 8000 is used to respond to HTTP requests and port 8001 is used to respond to GRPC requests by default. There are usually two types of resources that users request.

### **Model Metadata**

**HTTP**

Way to access: GET `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`

Use GET to request the url path to get model metadata in service, where `${MODEL_NAME}` indicates the name of the model and `${MODEL_VERSION}` indicates the version of the model. The server will return the metadata in json format as dictionary. With `$metadata_model_response` indicating the return object the content is as follows:

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

**GRPC**

The GRPC of the model service is defined as：

```text
service GRPCInferenceService
{
  // Check liveness of the inference server.
  rpc ServerLive(ServerLiveRequest) returns (ServerLiveResponse) {}

  // Check readiness of the inference server.
  rpc ServerReady(ServerReadyRequest) returns (ServerReadyResponse) {}

  // Check readiness of a model in the inference server.
  rpc ModelReady(ModelReadyRequest) returns (ModelReadyResponse) {}

  // Get server metadata.
  rpc ServerMetadata(ServerMetadataRequest) returns (ServerMetadataResponse) {}

  // Get model metadata.
  rpc ModelMetadata(ModelMetadataRequest) returns (ModelMetadataResponse) {}

  // Perform inference using a specific model.
  rpc ModelInfer(ModelInferRequest) returns (ModelInferResponse) {}
}
```

Way to access: Call ModelMetadata method defined in Model Service GRPC interface using GRPC client.

The structure of the requested ModelMetadataRequest message and the returned ServerMetadataResponse message in the interface is as follows, which is basically the same as the json structure above for HTTP.

```text
message ModelMetadataRequest
{
  // The name of the model.
  string name = 1;

  // The version of the model to check for readiness. If not given the
  // server will choose a version based on the model and internal policy.
  string version = 2;
}

message ModelMetadataResponse
{
  // Metadata for a tensor.
  message TensorMetadata
  {
    // The tensor name.
    string name = 1;

    // The tensor data type.
    string datatype = 2;

    // The tensor shape. A variable-size dimension is represented
    // by a -1 value.
    repeated int64 shape = 3;
  }

  // The model name.
  string name = 1;

  // The versions of the model available on the server.
  repeated string versions = 2;

  // The model's platform. See Platforms.
  string platform = 3;

  // The model's inputs.
  repeated TensorMetadata inputs = 4;

  // The model's outputs.
  repeated TensorMetadata outputs = 5;
}
```

### **Inference Service**

**HTTP**

Way to access: POST `v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer`

Use POST to request the url path to request the inference service of the model and get the inference result. Data in the POST request is also uploaded in json format. With `$inference_request` indicating uploaded objects, the content is as follows:
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
where `$tensor_data` represents a one-dimensional or multi-dimensional array. In the case of one-dimensional data, it should be arranged in row-major order.
After the server inference is completed, the result will be returned. With `$inference_response` representing return objects, the content is as follow:

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

**GRPC**

Way to access: Call ModelMetadata method defined in Model Service GRPC interface using GRPC client.

The structure of the requested ModelMetadataRequest message and the returned ModelInferResponse message in the interface is as follows, for full definition, please refer to [the GRPC part](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#grpc) in kserve Predict Protocol.

```text
message ModelInferRequest
{
  // An input tensor for an inference request.
  message InferInputTensor
  {
    // The tensor name.
    string name = 1;

    // The tensor data type.
    string datatype = 2;

    // The tensor shape.
    repeated int64 shape = 3;

    // Optional inference input tensor parameters.
    map<string, InferParameter> parameters = 4;

    // The tensor contents using a data-type format. This field must
    // not be specified if "raw" tensor contents are being used for
    // the inference request.
    InferTensorContents contents = 5;
  }

  // An output tensor requested for an inference request.
  message InferRequestedOutputTensor
  {
    // The tensor name.
    string name = 1;

    // Optional requested output tensor parameters.
    map<string, InferParameter> parameters = 2;
  }

  // The name of the model to use for inferencing.
  string model_name = 1;

  // The version of the model to use for inference. If not given the
  // server will choose a version based on the model and internal policy.
  string model_version = 2;

  // Optional identifier for the request. If specified will be
  // returned in the response.
  string id = 3;

  // Optional inference parameters.
  map<string, InferParameter> parameters = 4;

  // The input tensors for the inference.
  repeated InferInputTensor inputs = 5;

  // The requested output tensors for the inference. Optional, if not
  // specified all outputs produced by the model will be returned.
  repeated InferRequestedOutputTensor outputs = 6;

  // The data contained in an input tensor can be represented in "raw"
  // bytes form or in the repeated type that matches the tensor's data
  // type. To use the raw representation 'raw_input_contents' must be
  // initialized with data for each tensor in the same order as
  // 'inputs'. For each tensor, the size of this content must match
  // what is expected by the tensor's shape and data type. The raw
  // data must be the flattened, one-dimensional, row-major order of
  // the tensor elements without any stride or padding between the
  // elements. Note that the FP16 and BF16 data types must be represented as
  // raw content as there is no specific data type for a 16-bit float type.
  //
  // If this field is specified then InferInputTensor::contents must
  // not be specified for any input tensor.
  repeated bytes raw_input_contents = 7;
}

message ModelInferResponse
{
  // An output tensor returned for an inference request.
  message InferOutputTensor
  {
    // The tensor name.
    string name = 1;

    // The tensor data type.
    string datatype = 2;

    // The tensor shape.
    repeated int64 shape = 3;

    // Optional output tensor parameters.
    map<string, InferParameter> parameters = 4;

    // The tensor contents using a data-type format. This field must
    // not be specified if "raw" tensor contents are being used for
    // the inference response.
    InferTensorContents contents = 5;
  }

  // The name of the model used for inference.
  string model_name = 1;

  // The version of the model used for inference.
  string model_version = 2;

  // The id of the inference request if one was specified.
  string id = 3;

  // Optional inference response parameters.
  map<string, InferParameter> parameters = 4;

  // The output tensors holding inference results.
  repeated InferOutputTensor outputs = 5;

  // The data contained in an output tensor can be represented in
  // "raw" bytes form or in the repeated type that matches the
  // tensor's data type. To use the raw representation 'raw_output_contents'
  // must be initialized with data for each tensor in the same order as
  // 'outputs'. For each tensor, the size of this content must match
  // what is expected by the tensor's shape and data type. The raw
  // data must be the flattened, one-dimensional, row-major order of
  // the tensor elements without any stride or padding between the
  // elements. Note that the FP16 and BF16 data types must be represented as
  // raw content as there is no specific data type for a 16-bit float type.
  //
  // If this field is specified then InferOutputTensor::contents must
  // not be specified for any output tensor.
  repeated bytes raw_output_contents = 6;
}
```


## Client Tools

You can use HTTP client tool to request HTTP server or GRPC client tool for GRPC server once you get to know the interface provided by fastdeployserver. When fastdeployserver starts, port 8000 is used to respond to HTTP requests and port 8001 is used to respond to GRPC requests by default. Besides, you can use VisualDL as [fastdeploy client](#use-visualdl-as-fastdeploy-client-for-request-visualization) for request visualization.

### Using HTTP client

Here is how to use tritonclient and requests library to access fastdeployserver HTTP service. The first tool is a client specifically made for model service, which encapsulates request and response. And the second tool is a general http client tool, using which for access can help you better understand the above data structure.

1. Using tritonclient to access service

    Install tritonclient\[http\]

    ```bash
    pip install tritonclient[http]
    ```

   - Get metadata in YOLOv5 model
        ```python
        import tritonclient.http as httpclient  # Importing httpclient.
        server_addr = 'localhost:8000'  # Please change to the real address to fastdeployserver server here.
        client = httpclient.InferenceServerClient(server_addr)  # Create clients.
        model_metadata = client.get_model_metadata(
            model_name='yolov5', model_version='1') # Request metadata in YOLOv5 model.
        ```
        You can print model's input and output.
        ```python
        print(model_metadata.inputs)
        ```

        ```text
        [{'name': 'INPUT', 'datatype': 'UINT8', 'shape': [-1, -1, -1, 3]}]
        ```

        ```python
        print(model_metadata.outputs)
        ```

        ```text
        [{'name': 'detction_result', 'datatype': 'BYTES', 'shape': [-1, -1]}]
        ```

    - Request Inference Service

        You can create data according to inputs and outputs of the model, and the request inference.

        ```python
        # Assume that the file name of image data is 000000014439.jpg.
        import cv2
        image = cv2.imread('000000014439.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]

        inputs = []
        infer_input = httpclient.InferInput('INPUT', image.shape, 'UINT8') # Create inputs.
        infer_input.set_data_from_numpy(image)  # Load input data.
        inputs.append(infer_input)
        outputs = []
        infer_output = httpclient.InferRequestedOutput('detction_result') # Create outputs.
        outputs.append(infer_output)
        response = client.infer(
                    'yolov5', inputs, model_version='1', outputs=outputs)  # Request inference.
        response_outputs = response.as_numpy('detction_result') # Get results based on output variable name.
        ```

2.  Using requests to access service

    Install requests
    ```bash
    pip install requests
    ```
    - Get metadata in YOLOv5 model

        ```python
        import requests
        url = 'http://localhost:8000/v2/models/yolov5/versions/1' # Construct the url based on "Model Metadata" in the above section.
        response = requests.get(url)
        response = response.json() # Return data as json, and parse in json format.
        ```
        Print the metadata returned.
        ```python
        print(response)
        ```
        ```text
        {'name': 'yolov5', 'versions': ['1'], 'platform': 'ensemble', 'inputs': [{'name': 'INPUT', 'datatype': 'UINT8', 'shape': [-1, -1, -1, 3]}], 'outputs': [{'name': 'detction_result', 'datatype': 'BYTES', 'shape': [-1, -1]}]}
        ```
    - Request Inference Service

        You can create data according to inputs and outputs of the model, and the request inference.
        ```python
        url = 'http://localhost:8000/v2/models/yolov5/versions/1/infer' # Construct the url based on "Inference Service" in the above section.
        # Assume that the file name of image data is 000000014439.jpg.
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
        response = response.json()  # Return data as json, parse in json format, and you get your inference result.
        ```

### Using GRPC client

Install tritonclient\[grpc\]
```bash
pip install tritonclient[grpc]
```
Tritonclient\[grpc\] provides a client using GRPC and encapsulates the interaction of GRPC. So you do not have to establish a connection with the server manually or use the stub in grpc to call the server interface directly, but to use the same interface as the tritonclient HTTP client.

- Get metadata in YOLOv5 model
    ```python
    import tritonclient.grpc as grpcclient # Import grpc client.
    server_addr = 'localhost:8001'  # Please change to the real address to fastdeployserver grpc server here.
    client = grpcclient.InferenceServerClient(server_addr)  # Create clients
    model_metadata = client.get_model_metadata(
        model_name='yolov5', model_version='1') # Request metadata in YOLOv5 model.
    ```
- Request Inference Service
  Create request data according to the returned model_metadata. Let us first print the input and output.
    ```python
    print(model_metadata.inputs)
    ```
    ```text
    [name: "INPUT"
    datatype: "UINT8"
    shape: -1
    shape: -1
    shape: -1
    shape: 3
    ]
    ```

    ```python
    print(model_metadata.outputs)
    ```

    ```text
    [name: "detction_result"
    datatype: "BYTES"
    shape: -1
    shape: -1
    ]
    ```

    Create data according to inputs and outputs, and then request inference.
    ```python
    # Assume that the file name of image data is 000000014439.jpg.
    import cv2
    image = cv2.imread('000000014439.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[None]

    inputs = []
    infer_input = grpcclient.InferInput('INPUT', image.shape, 'UINT8') # Create inputs.
    infer_input.set_data_from_numpy(image)  # Load input data.
    inputs.append(infer_input)
    outputs = []
    infer_output = grpcclient.InferRequestedOutput('detction_result') # Create outputs.
    outputs.append(infer_output)
    response = client.infer(
                'yolov5', inputs, model_version='1', outputs=outputs)  # Request inference
    response_outputs = response.as_numpy('detction_result') # Get results based on output variable name.
    ```


### Use VisualDL as fastdeploy client for request visualization

The FastDeploy Client component is mainly used to quickly access the fastdeployserver service, to help users visualize prediction requests and results, and make quick verification of deployed services. This component is developed based on the gradio.
  <p align="center">
   <img src="https://user-images.githubusercontent.com/22424850/211204267-8e044f32-1008-46a7-828a-d7c27ac5754a.gif" width="100%"/>
   </p>

#### Usage

Install VisualDL (version>=2.5.0)

```shell
python3 -m pip install visualdl
```

use command

```shell
visualdl --host 0.0.0.0 --port 8080
```

Then open `http://127.0.0.1:8080` in the browser (please replace it with the ip of the machine that starts visualdl if not localhost), you can see the component tab of FastDeploy Client.

#### Function Description

The client component is mainly divided into four parts. The first part is the parameter setting area of the fastdeployserver service, including the ip and port number of the http service, the port number of the metrics service, the model name and version number of the http inference service to be requested. The second part is the input and output area of the model, which helps users visualize prediction requests and results. The third part is the metrics area of the service, which is used to display the current performance metrics of the service. The fourth part is used to display the execution status of each operation.

  <p align="center">
   <img src="https://user-images.githubusercontent.com/22424850/211206389-9932d606-ea71-4f05-87eb-dcc21c5eeec9.png" width="100%"/>
   </p>

- Parameter setting area of fastdeployserver service

   Set the ip address and port information of the server to be requested, as well as the model name and version number to perform inference. Subsequent requests will be sent to the set address.


- Input and output areas of the model

   Currently, there are two ways to access the service. The first one is "component form". This method will directly obtain the input and output of the model deployed on the server, and represent input and output variables by the components of Gradio. Each input and output variable is equipped with a text component and an image component. The user selects the corresponding component to use according to the actual type of the variable. For example, if the variable is image data, use the image component to input it, and if it is text, use the text component to input it. The returned data is also visualized through components. Since the visualization operations of different tasks are different, the task type selection is provided above. When the task type is not specified, the output variable is only displayed with the text component, displaying the original data returned by the server. When the task type is specified, the returned data will be parsed and displayed visually using the image component.
   <p align="center">
   <img src="https://user-images.githubusercontent.com/22424850/211207902-4717011d-8ae2-4105-9508-ab164896f177.gif" width="100%"/>
   </p>


   The second is "original form", which is equivalent to an original http client, where the payload (json body) of the http request is input in the input text box, and the original payload returned by the server is displayed in the output text box. For example, request payload for [paddledetection](../../../examples/vision/detection/paddledetection/serving/README.md) is as follows, complete content can be found in [file](../../../examples/vision/detection/paddledetection/serving/ppdet_request.json).

   ```json
  {
  "inputs": [
    {
      "name": "INPUT",
      "shape": [1, 404, 640, 3],
      "datatype": "UINT8",
      "data": image content
    },
  "outputs": [
    {
      "name": "DET_RESULT"
    }
  ]
  }
  ```


   <p align="center">
   <img src="https://user-images.githubusercontent.com/22424850/211208731-381222bb-8fbe-45fa-bf78-4a3e2c7f6f04.gif" width="100%"/>
   </p>


- Service metrics area

   It is used to display the performance information returned by the fastdeployserver metrics service, including the execution statistics for  each request, response delay, and the utilization rate of computing devices in the environment.

   <p align="center">
   <img src="https://user-images.githubusercontent.com/22424850/211208071-e772ed55-9a3d-4a21-9bca-3d80f444ca64.gif" width="100%"/>
   </p>
- Execution status display area

   Displays the status of all operations performed on the client component. When the user clicks the "get model input and output", "submit request" and "update metrics" buttons, if an exception occurs, the execution status display area will show the exception messages. And if the execution is successful, there will be a corresponding tips so that the user can know the execution successes.
