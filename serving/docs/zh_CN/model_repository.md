中文 ｜ [English](../EN/model_repository-en.md)
# 模型仓库(Model Repository)

FastDeploy启动服务时指定模型仓库中一个或多个模型部署服务。当服务运行时，可以用[Model Management](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md)中描述的方式修改服务中的模型。
从服务器启动时指定的一个或多个模型存储库中为模型提供服务

## 仓库结构
模型仓库路径通过FastDeploy启动时的*--model-repository*选项指定，可以多次指定*--model-repository*选项来加载多个仓库。例如:

```
$ fastdeploy --model-repository=<model-repository-path>
```

模型仓库的结构必须按以下的格式创建:
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
在最顶层`<model-repository-path>`模型仓库目录下，必须有0个或多个`<model-name>`模型名字的子目录。每个`<model-name>`模型名字子目录包含部署模型相应的信息，多个表示模型版本的数字子目录和一个描述模型配置的*config.pbtxt*文件。

Paddle模型存在版本号子目录中，必须为`model.pdmodel`文件和`model.pdiparams`文件。

## 模型版本
每个模型在仓库中可以有一个或多个可用的版本，模型目录中以数字命名的子目录就是对应的版本，数字即版本号。没有以数字命名的子目录，或以*0*开头的子目录都会被忽略。模型配置文件中可以指定[版本策略](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#version-policy)，控制Triton启动模型目录中的哪个版本。

## 模型仓库示例
部署Paddle模型时需要的模型必须是2.0版本以上导出的推理模型，模型包含`model.pdmodel`和`model.pdiparams`两个文件放在版本目录中。

部署Paddle模型的最小模型仓库目录示例:
```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.pdmodel
        model.pdiparams

  # 真实例子:
  models
  └── ResNet50
      ├── 1
      │   ├── model.pdiparams
      │   └── model.pdmodel
      └── config.pbtxt
```

部署ONNX模型，必须要在版本目录中包含`model.onnx`名字的模型。

部署ONNX模型的最小模型仓库目录示例:
```
  <model-repository-path>/
    <model-name>/
      config.pbtxt
      1/
        model.onnx

  # 真实例子:
  models
  └── ResNet50
      ├── 1
      │   ├── model.onnx
      └── config.pbtxt
```
