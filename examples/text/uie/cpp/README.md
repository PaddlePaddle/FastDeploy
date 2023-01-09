English | [简体中文](README_CN.md) 

# Universal Information Extraction UIE C++ Deployment Example

This directory provides `infer.cc` quickly complete the example on CPU/GPU by [UIE Model](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)

Before deployment, two steps need to be confirmed.

- 1. The software and hardware environment meets the requirements. Please refer to [Environment requirements for FastDeploy](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. Download precompiled deployment library and samples code based on the develop environment. Please refer to [FastDeploy pre-compiled libraries](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

## A Quick Start
Take uie-base model inference on Linux as an example, execute the following command in this directory to complete the compilation test. FastDeploy version 0.7.0 or above is required to support this model (x.x.x>=0.7.0).

```
mkdir build
cd build
# Download FastDeploy precompiled library. Users can choose proper versions in the `FastDeploy pre-compiled libraries` mentioned above.
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the uie-base model and vocabulary
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz


# CPU Inference
./infer_demo uie-base 0

# GPU Inference
./infer_demo uie-base 1

# Use OpenVINO for inference
./infer_demo uie-base 1 2
```

The results after running are as follows (only the output of the NER task is captured).
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

## The way to use the UIE model in each extraction task

In the UIE model, schema represents the structured information to be extracted, so the UIE model can support different information extraction tasks by setting different schemas.

### Initialize UIEModel

```c++
std::string model_dir = "uie-base";
std::string model_path = model_dir + sep + "inference.pdmodel";
std::string param_path = model_dir + sep + "inference.pdiparams";
std::string vocab_path = model_dir + sep + "vocab.txt";
using fastdeploy::text::SchemaNode;
using fastdeploy::text::UIEResult;
// Define the uie result object
std::vector<std::unordered_map<std::string, std::vector<UIEResult>>> results;

// Initialize UIE model
auto predictor =
    fastdeploy::text::UIEModel(model_path, param_path, vocab_path, 0.5, 128,
                                {"时间", "选手", "赛事名称"}, option);
```

### Entity Extraction

The initialization stage sets the schema```["time", "player", "event name"]``` to extract the time, player and event name from the input text.

```c++
// Named Entity Recognition
predictor.Predict({"2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷"
                    "爱凌以188.25分获得金牌！"},
                &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 赛事名称:
//     text: 北京冬奥会自由式滑雪女子大跳台决赛
//     probability: 0.850309
//     start: 6
//     end: 23
//
// 时间:
//     text: 2月8日上午
//     probability: 0.985738
//     start: 0
//     end: 6
//
// 选手:
//     text: 谷爱凌
//     probability: 0.898155
//     start: 28
//     end: 31
```

For example, if the target entity types are "肿瘤的大小", "肿瘤的个数", "肝癌级别" and "脉管内癌栓分级", the following statements can be executed.

```c++
predictor.SetSchema(
    {"肿瘤的大小", "肿瘤的个数", "肝癌级别", "脉管内癌栓分级"});
predictor.Predict({"（右肝肿瘤）肝细胞性肝癌（II-"
                    "III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵"
                    "及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形"
                    "成。（肿物1个，大小4.2×4.0×2.8cm）。"},
                &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 脉管内癌栓分级:
//     text: M0级
//     probability: 0.908329
//     start: 67
//     end: 70
//
// 肝癌级别:
//     text: II-III级
//     probability: 0.924327
//     start: 13
//     end: 20
//
// 肿瘤的大小:
//     text: 4.2×4.0×2.8cm
//     probability: 0.834113
//     start: 87
//     end: 100
//
// 肿瘤的个数:
//     text: 1个
//     probability: 0.753841
//     start: 82
//     end: 84

```

### Relation Extraction

Relation Extraction (RE) refers to identifying entities from text and extracting semantic relationships between them to obtain triadic information, i.e. <subject, predicate, object>.

For example, if we take "contest name" as the extracted entity, and the relations are "主办方", "承办方" and "已举办次数", then we can write the following statements.

```c++
predictor.SetSchema(
    {SchemaNode("竞赛名称", {SchemaNode("主办方"), SchemaNode("承办方"),
                            SchemaNode("已举办次数")})});
predictor.Predict(
    {"2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度"
    "公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会"
    "承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"},
    &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 竞赛名称:
//     text: 2022语言与智能技术竞赛
//     probability: 0.78254
//     start: 0
//     end: 13
//     relation:
//         已举办次数:
//             text: 4届
//             probability: 0.46713
//             start: 80
//             end: 82
//
//         主办方:
//             text: 中国中文信息学会
//             probability: 0.842172
//             start: 14
//             end: 22
//
//             text: 中国计算机学会
//             probability: 0.758081
//             start: 23
//             end: 30
//
//         承办方:
//             text: 百度公司
//             probability: 0.829271
//             start: 35
//             end: 39
//
//             text: 中国中文信息学会评测工作委员会
//             probability: 0.70005
//             start: 40
//             end: 55
//
//             text: 中国计算机学会自然语言处理专委会
//             probability: 0.619348
//             start: 56
//             end: 72
```

### Event Extraction

Event Extraction (EE) refers to extracting predefined Trigger and Argument from natural language texts and combining them into structured event information.

For example, if the targets are"地震强度", "时间", "震中位置" and "引源深度" for the event "地震", we can execute the following codes.

```c++
predictor.SetSchema({SchemaNode(
    "地震触发词", {SchemaNode("地震强度"), SchemaNode("时间"),
                    SchemaNode("震中位置"), SchemaNode("震源深度")})});
predictor.Predict(
    {"中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24."
    "34度，东经99.98度)发生3.5级地震，震源深度10千米。"},
    &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 地震触发词:
//     text: 地震
//     probability: 0.997743
//     start: 56
//     end: 58
//     relation:
//         震源深度:
//             text: 10千米
//             probability: 0.993797
//             start: 63
//             end: 67
//
//         震中位置:
//             text: 云南临沧市凤庆县(北纬24.34度，东经99.98度)
//             probability: 0.787402
//             start: 23
//             end: 50
//
//         地震强度:
//             text: 3.5级
//             probability: 0.99808
//             start: 52
//             end: 56
//
//         时间:
//             text: 5月16日06时08分
//             probability: 0.98533
//             start: 11
//             end: 22
```

### Opinion Extraction

opinion extraction refers to the extraction of evaluation dimensions and opinions contained in the text.

For example, if the extraction target is the evaluation dimensions and their corresponding opinions and sentiment tendencies. We can execute the following codes：

```c++
predictor.SetSchema({SchemaNode(
    "评价维度",
    // NOTE(zhoushunjie): It's necessary to explicitly use
    // std::vector to convert initializer list of SchemaNode whose size is
    // two. If not to do so, an ambiguous compliation error will occur in
    // mac x64 platform.
    std::vector<SchemaNode>{SchemaNode("观点词"),
                            SchemaNode("情感倾向[正向，负向]")})});
predictor.Predict(
    {"店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"},
    &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 评价维度:
//     text: 店面
//     probability: 0.969685
//     start: 0
//     end: 2
//     relation:
//         情感倾向[正向，负向]:
//             text: 正向
//             probability: 0.998215
//
//         观点词:
//             text: 干净
//             probability: 0.994532
//             start: 2
//             end: 4
//
//     text: 性价比
//     probability: 0.981704
//     start: 17
//     end: 20
//     relation:
//         情感倾向[正向，负向]:
//             text: 正向
//             probability: 0.996614
//
//         观点词:
//             text: 高
//             probability: 0.957397
//             start: 21
//             end: 22
```

### Sentiment Classification

Sentence-level sentiment classification, i.e., determining a sentence has a "positive" sentiment or "negative" sentiment. We can execute the following codes:


```c++
predictor.SetSchema(SchemaNode("情感倾向[正向，负向]"));
predictor.Predict({"这个产品用起来真的很流畅，我非常喜欢"}, &results);
std::cout << results << std::endl;
results.clear();

// An output example
// The result:
// 情感倾向[正向，负向]:
//     text: 正向
//     probability: 0.999002
```

### Cross-task Extraction

or example, in a legal scenario where both entity extraction and relation extraction need to be performed. We can execute the following codes:


```c++
predictor.SetSchema({SchemaNode("法院", {}),
                    SchemaNode("原告", {SchemaNode("委托代理人")}),
                    SchemaNode("被告", {SchemaNode("委托代理人")})});
predictor.Predict({"北京市海淀区人民法院\n民事判决书\n(199x)"
                    "建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 "
                    "A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司"
                    "总经理。\n委托代理人赵六，北京市 C律师事务所律师。"},
                &results);
std::cout << results << std::endl;
results.clear();
// An output example
// The result:
// 被告:
//     text: B公司
//     probability: 0.843735
//     start: 64
//     end: 67
//     relation:
//         委托代理人:
//             text: 赵六
//             probability: 0.726712
//             start: 90
//             end: 92
//
// 法院:
//     text: 北京市海淀区人民法院
//     probability: 0.922107
//     start: 0
//     end: 10
//
// 原告:
//     text: 张三
//     probability: 0.994981
//     start: 35
//     end: 37
//     relation:
//         委托代理人:
//             text: 李四
//             probability: 0.795686
//             start: 44
//             end: 46
```

## UIEModel C++ Interface

### SchemaNode Structure
Represent the structure of UIE model target mode.

```c++
SchemaNode(const std::string& name,
           const std::vector<SchemaNode>& children = {});
```
**Parameter**

> * **name**(str): information requiring extraction.
> * **children**(str): the current node needs to extract the sub-information associated with the original information.

### UIEModel Structure
The UIE model structure for information extraction task.

#### Initialized Function
```c++
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const std::vector<std::string>& schema,
    const fastdeploy::RuntimeOption& custom_option =
        fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format = fastdeploy::ModelFormat::PADDLE,
    SchemaLanguage schema_language = SchemaLanguage::ZH);
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const SchemaNode& schema, const fastdeploy::RuntimeOption& custom_option =
                                  fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format = fastdeploy::ModelFormat::PADDLE,
    SchemaLanguage schema_language = SchemaLanguage::ZH);
UIEModel(
    const std::string& model_file, const std::string& params_file,
    const std::string& vocab_file, float position_prob, size_t max_length,
    const std::vector<SchemaNode>& schema,
    const fastdeploy::RuntimeOption& custom_option =
        fastdeploy::RuntimeOption(),
    const fastdeploy::ModelFormat& model_format =
        fastdeploy::ModelFormat::PADDLE,
    SchemaLanguage schema_language = SchemaLanguage::ZH);
```

UIEModel loading and initialization. Among them, model_file, params_file are Paddle inference documents exported by trained models. Please refer to [Model export](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2).

**Parameter**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path
> * **vocab_file**(str):  Vocabulary file
> * **position_prob**(str): Position probability. The model will output positions with probability greater than `position_prob`, default is 0.5
> * **max_length**(int):  Maximized length of input text. Input text subscript exceeding `max_length` will be truncated. Default is 128
> * **schema**(list(SchemaNode) | SchemaNode | list(str)): Target information for extraction tasks
> * **runtime_option**(RuntimeOption): Backend inference configuration, the default is None, i.e., the default configuration
> * **model_format**(ModelFormat): Model format, and default is Paddle format
> * **schema_language** (SchemaLanguage): Schema language, and default is ZH（Chinese）. Currently supported language：ZH（Chinese），EN（English）

#### SetSchema Function

```c++
void SetSchema(const std::vector<std::string>& schema);
void SetSchema(const std::vector<SchemaNode>& schema);
void SetSchema(const SchemaNode& schema);
```

**Parameter**
> * **schema**(list(SchemaNode) | SchemaNode | list(str)): Input data, in a text pattern to be extracted.

#### Predict Function

```c++
void Predict(
    const std::vector<std::string>& texts,
    std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>* results);
```
**Parameter**

> * **texts**(list(str)): text list
> * **results**(list(dict())): UIE model extraction results

## Related Documents

[Details for UIE model](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md)

[How to export a UIE model](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)

[UIE Python deployment](../python/README.md)
