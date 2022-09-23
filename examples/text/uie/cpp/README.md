# 通用信息抽取 UIE C++部署示例

本目录下提供`infer.cc`快速完成[UIE模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)在CPU/GPU的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/environment.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/quick_start)

## 快速开始
以Linux上uie-base模型推理为例，在本目录执行如下命令即可完成编译测试。

```
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.2.1.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.1.tgz

cd fastdeploy-linux-x64-gpu-0.2.1/examples/text/uie/cpp
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../fastdeploy-linux-x64-gpu-0.2.1
make -j

# 下载uie-base模型以及词表
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz


# CPU 推理
./infer_demo uie-base 0

# GPU 推理
./infer_demo uie-base 1

# 使用OpenVINO推理
./infer_demo uie-base 1 2
```

运行完成后返回结果如下所示(仅截取NER任务的输出)。
```bash
[INFO] fastdeploy/fastdeploy_runtime.cc(264)::Init      Runtime initialized with Backend::PDINFER in device Device::CPU.
After init predictor
The result:
赛事名称:
    text: 北京冬奥会自由式滑雪女子大跳台决赛
    probability: 0.850309
    start: 6
    end: 23

时间:
    text: 2月8日上午
    probability: 0.985738
    start: 0
    end: 6

选手:
    text: 谷爱凌
    probability: 0.898155
    start: 28
    end: 31
```

## UIEModel C++接口

### SchemaNode 结构
表示UIE模型目标模式的结构。

```c++
SchemaNode(const std::string& name,
           const std::vector<SchemaNode>& children = {});
```
**参数**

> * **name**(str): 需要抽取的信息。
> * **children**(str): 当前节点需抽取信息关联的子信息。

### UIEModel 结构
用于信息抽取任务的UIE模型结构。

#### 初始化函数
```c++
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const std::vector<std::string>& schema,
    const fastdeploy::RuntimeOption& custom_option =
        fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format = fastdeploy::ModelFormat::PADDLE);
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const SchemaNode& schema, const fastdeploy::RuntimeOption& custom_option =
                                  fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format = fastdeploy::ModelFormat::PADDLE);
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const std::vector<SchemaNode>& schema,
    const fastdeploy::RuntimeOption& custom_option =
        fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format = fastdeploy::ModelFormat::PADDLE);
```

UIE模型加载和初始化，其中model_file, params_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **vocab_file**(str): 词表文件路径
> * **position_prob**(str): 位置概率，模型将输出位置概率大于`position_prob`的位置，默认为0.5
> * **max_length**(int): 输入文本的最大长度。输入文本下标超过`max_length`的部分将被截断。默认为128
> * **schema**(list(SchemaNode) | SchemaNode | list(str)): 抽取任务的目标模式。
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

#### SetSchema函数

```c++
void SetSchema(const std::vector<std::string>& schema);
void SetSchema(const std::vector<SchemaNode>& schema);
void SetSchema(const SchemaNode& schema);
```

**参数**
> * **schema**(list(SchemaNode) | SchemaNode | list(str)): 输入数据，待抽取文本模式。

#### Predict函数

```c++
void Predict(
    const std::vector<std::string>& texts,
    std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>* results);
```
**参数**

> * **texts**(list(str)): 文本列表
> * **results**(list(dict())): UIE模型抽取结果。UIEResult结构详细可见[UIEResult说明](../../../../docs/api/text_results/uie_result.md)。

## 相关文档

[UIE模型详细介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md)

[UIE模型导出方法](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)

[UIE C++部署方法](../cpp/README.md)
