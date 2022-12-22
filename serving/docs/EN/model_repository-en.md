English | [中文](../zh_CN/model_repository.md)
# Model Repository

FastDeploy starts the serving by specifying one or more models in the model repository to deploy the service. When the serving is running, the models in the service can be modified following [Model Management](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md), and obtain serving from one or more model repositories specified at the serving initiation.

## Repository Architecture

The model repository path is specified via the *--model-repository* option at FastDeploy's initation, and multiple repositories can be loaded by specifying the *--model-repository* option multiple times. Example:

```
$ fastdeploy --model-repository=<model-repository-path>
```

Model repository architecture should comply the following format:

```
  <model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

At the topmost `<model-repository-path>` model repository directory, there must be 0 or more `<model-name>` subdirectories. Each `<model-name>` subdirectory contains information corresponding to the model deployment, multiple numeric subdirectories indicating the model version, and a *config.pbtxt* file describing the model configuration.

Paddle models are saved in the version number subdirectory, which must be `model.pdmodel`  and `model.pdiparams` files.

## Model Version

Each model can have one or more versions available in the repository. The subdirectory named with a number in the model directory implies the version number. Subdirectories that are not named with a number, or that start with *0* will be ignored. A [version policy](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#version-policy) can be specified in the model configuration file to control which version of the model in model  directory is launched by Triton.

## Repository Demo

The model needed for Paddle deployment must be an inference model exported from version 2.0 or higher. The model contains `model.pdmodel` and `model.pdiparams`  in the version directory.

Example: A minimal model repository directory for deploying Paddle models

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pdmodel
        model.pdiparams

  # Example:
  models
  └── ResNet50
      ├── 1
      │   ├── model.pdiparams
      │   └── model.pdmodel
      └── config.pbtxt
```

To deploy an ONNX model, model with the name `model.onnx` must be included in the version directory

Example: A minimal model repository directory for deploying ONNX models

```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx

  # Example:
  models
  └── ResNet50
      ├── 1
      │   ├── model.onnx
      └── config.pbtxt
```
