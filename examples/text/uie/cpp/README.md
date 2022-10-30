# 通用信息抽取 UIE C++部署示例

本目录下提供`infer.cc`快速完成[UIE模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)在CPU/GPU的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/environment.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/quick_start)

## 快速开始
以Linux上uie-base模型推理为例，在本目录执行如下命令即可完成编译测试。

```
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.4.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.4.0.tgz

cd fastdeploy-linux-x64-gpu-0.4.0/examples/text/uie/cpp
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../../fastdeploy-linux-x64-gpu-0.4.0
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

## UIE模型各抽取任务使用方式

在UIE模型中，schema代表要抽取的结构化信息，所以UIE模型可通过设置不同的schema支持不同信息抽取任务。

### 初始化UIEModel

```c++
std::string model_dir = "uie-base";
std::string model_path = model_dir + sep + "inference.pdmodel";
std::string param_path = model_dir + sep + "inference.pdiparams";
std::string vocab_path = model_dir + sep + "vocab.txt";
using fastdeploy::text::SchemaNode;
using fastdeploy::text::UIEResult;
// 定义uie result对象
std::vector<std::unordered_map<std::string, std::vector<UIEResult>>> results;

// 初始化UIE模型
auto predictor =
    fastdeploy::text::UIEModel(model_path, param_path, vocab_path, 0.5, 128,
                                {"时间", "选手", "赛事名称"}, option);
```

### 实体抽取

初始化阶段将schema设置为```["时间", "选手", "赛事名称"]```，可对输入的文本抽取时间、选手以及赛事名称三个信息。

```c++
// Named Entity Recognition
predictor.Predict({"2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷"
                    "爱凌以188.25分获得金牌！"},
                &results);
std::cout << results << std::endl;
results.clear();

// 示例输出
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

例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", 则可编写如下语句：

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

// 示例输出
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

### 关系抽取

关系抽取（Relation Extraction，简称RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"已举办次数", 则可编写如下语句：

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

// 示例输出
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

### 事件抽取

事件抽取 (Event Extraction, 简称EE)，是指从自然语言文本中抽取预定义的事件触发词(Trigger)和事件论元(Argument)，组合为相应的事件结构化信息。

例如抽取的目标是"地震"事件的"地震强度"、"时间"、"震中位置"和"震源深度"这些信息，则可编写如下代码：

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

// 示例输出
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

### 评论观点抽取

评论观点抽取，是指抽取文本中包含的评价维度、观点词。

例如抽取的目标是文本中包含的评价维度及其对应的观点词和情感倾向，可编写以下代码：

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

// 示例输出
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

### 情感分类

句子级情感倾向分类，即判断句子的情感倾向是“正向”还是“负向”，可编写以下代码：

```c++
predictor.SetSchema(SchemaNode("情感倾向[正向，负向]"));
predictor.Predict({"这个产品用起来真的很流畅，我非常喜欢"}, &results);
std::cout << results << std::endl;
results.clear();

// 示例输出
// The result:
// 情感倾向[正向，负向]:
//     text: 正向
//     probability: 0.999002
```

### 跨任务抽取

例如在法律场景同时对文本进行实体抽取和关系抽取，可编写以下代码：

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
// 示例输出
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
> * **schema_language** (SchemaLanguage): Schema 语言，默认为ZH（中文），目前支持的语言种类包括：ZH（中文），EN（英文）。

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

[UIE Python部署方法](../python/README.md)
