English | [简体中文](README_CN.md)

# Universal Information Extraction UIE Python Deployment Example

Before deployment, two steps need to be confirmed.

- 1. The software and hardware environment meets the requirements. Please refer to [Environment requirements for FastDeploy](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. FastDeploy Python whl pacakage needs installation. Please refer to [FastDeploy Python Installation](../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

This directory provides an example that `infer.py` quickly complete CPU deployment conducted by the UIE model with OpenVINO acceleration on CPU/GPU and CPU.

## A Quick Start
```bash

# Download deployment sample code
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/text/uie/python

# Download the UIE model file and word list. Taking the uie-base model as an example.
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz

# CPU Inference
python infer.py --model_dir uie-base --device cpu
# GPU Inference
python infer.py --model_dir uie-base --device gpu
# Use OpenVINO for inference
python infer.py --model_dir uie-base --device cpu --backend openvino --cpu_num_threads 8
```

The results after running are as follows(only the output of the NER task is captured).
```bash
1. Named Entity Recognition Task
The extraction schema: ['时间', '选手', '赛事名称']
[{'时间': {'end': 6,
         'probability': 0.9857379794120789,
         'start': 0,
         'text': '2月8日上午'},
  '赛事名称': {'end': 23,
           'probability': 0.8503087162971497,
           'start': 6,
           'text': '北京冬奥会自由式滑雪女子大跳台决赛'},
  '选手': {'end': 31,
         'probability': 0.8981553912162781,
         'start': 28,
         'text': '谷爱凌'}}]

The extraction schema: ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
[{'肝癌级别': {'end': 20,
           'probability': 0.9243271350860596,
           'start': 13,
           'text': 'II-III级'},
  '肿瘤的个数': {'end': 84,
            'probability': 0.7538408041000366,
            'start': 82,
            'text': '1个'},
  '肿瘤的大小': {'end': 100,
            'probability': 0.8341134190559387,
            'start': 87,
            'text': '4.2×4.0×2.8cm'},
  '脉管内癌栓分级': {'end': 70,
              'probability': 0.9083293080329895,
              'start': 67,
              'text': 'M0级'}}]
......
```

### Description of command line arguments

`infer.py` supports more command line parameters than the above example. The following is a description of each command line parameter.

| Argument | Description |
|----------|--------------|
|--model_dir | The specified directory of model. |
|--batch_size | The batch size of inputs. |
|--max_length | The max length of sequence. Default to 128|
|--device | The device of runtime, choices: ['cpu', 'gpu']. Default to 'cpu' |
|--backend | The backend of runtime, choices: ['onnx_runtime', 'paddle_inference', 'openvino', 'tensorrt', 'paddle_tensorrt']. Default to 'paddle_inference'. |
|--use_fp16 | Whether to use fp16 precision to infer. It can be turned on when 'tensorrt' or 'paddle_tensorrt' backend is selected. Default to False.|

## The way to use the UIE model in each extraction task

In the UIE model, schema represents the structured information to be extracted, so the UIE model can support different information extraction tasks by setting different schemas.

### Initialize UIEModel

```python
import fastdeploy
from fastdeploy.text import UIEModel
model_dir = "uie-base"
model_path = os.path.join(model_dir, "inference.pdmodel")
param_path = os.path.join(model_dir, "inference.pdiparams")
vocab_path = os.path.join(model_dir, "vocab.txt")

runtime_option = fastdeploy.RuntimeOption()
schema = ["时间", "选手", "赛事名称"]

# Initialise UIE model
uie = UIEModel(
    model_path,
    param_path,
    vocab_path,
    position_prob=0.5,
    max_length=128,
    schema=schema,
    runtime_option=runtime_option)
```

### Entity Extraction

The initialization stage sets the schema```["time", "player", "event name"]``` to extract the time, player and event name from the input text.

```python
>>> from pprint import pprint
>>> results = uie.predict(
        ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"], return_dict=True)
>>> pprint(results)

# An output example
# [{'时间': {'end': 6,
#          'probability': 0.9857379794120789,
#          'start': 0,
#          'text': '2月8日上午'},
#   '赛事名称': {'end': 23,
#            'probability': 0.8503087162971497,
#            'start': 6,
#            'text': '北京冬奥会自由式滑雪女子大跳台决赛'},
#   '选手': {'end': 31,
#          'probability': 0.8981553912162781,
#          'start': 28,
#          'text': '谷爱凌'}}]

```

For example, if the target entity types are "肿瘤的大小", "肿瘤的个数", "肝癌级别" and "脉管内癌栓分级", the following statements can be executed.

```python
>>> schema = ["肿瘤的大小", "肿瘤的个数", "肝癌级别", "脉管内癌栓分级"]
>>> uie.set_schema(schema)
>>> results = uie.predict(
    [
        "（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，"
        "未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"
    ],
    return_dict=True)
>>> pprint(results)

# An output example
# [{'肝癌级别': {'end': 20,
#            'probability': 0.9243271350860596,
#            'start': 13,
#            'text': 'II-III级'},
#   '肿瘤的个数': {'end': 84,
#             'probability': 0.7538408041000366,
#             'start': 82,
#             'text': '1个'},
#   '肿瘤的大小': {'end': 100,
#             'probability': 0.8341134190559387,
#             'start': 87,
#             'text': '4.2×4.0×2.8cm'},
#   '脉管内癌栓分级': {'end': 70,
#               'probability': 0.9083293080329895,
#               'start': 67,
#               'text': 'M0级'}}]
```


### Relation Extraction

Relation Extraction (RE) refers to identifying entities from text and extracting semantic relationships between them to obtain triadic information, i.e. <subject, predicate, object>.

For example, if we take "contest name" as the extracted entity, and the relations are "主办方", "承办方" and "已举办次数", then we can execute the following statements.
```python
>>> schema = {"竞赛名称": ["主办方", "承办方", "已举办次数"]}
>>> uie.set_schema(schema)
>>> results = uie.predict(
    [
        "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作"
        "委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"
    ],
    return_dict=True)
>>> pprint(results)

# An output example
# [{'竞赛名称': {'end': 13,
#            'probability': 0.7825401425361633,
#            'relation': {'主办方': [{'end': 22,
#                                  'probability': 0.8421716690063477,
#                                  'start': 14,
#                                  'text': '中国中文信息学会'},
#                                 {'end': 30,
#                                  'probability': 0.7580805420875549,
#                                  'start': 23,
#                                  'text': '中国计算机学会'}],
#                         '已举办次数': [{'end': 82,
#                                    'probability': 0.4671304225921631,
#                                    'start': 80,
#                                    'text': '4届'}],
#                         '承办方': [{'end': 39,
#                                  'probability': 0.8292709589004517,
#                                  'start': 35,
#                                  'text': '百度公司'},
#                                 {'end': 55,
#                                  'probability': 0.7000502943992615,
#                                  'start': 40,
#                                  'text': '中国中文信息学会评测工作委员会'},
#                                 {'end': 72,
#                                  'probability': 0.6193484663963318,
#                                  'start': 56,
#                                  'text': '中国计算机学会自然语言处理专委会'}]},
#            'start': 0,
#            'text': '2022语言与智能技术竞赛'}}]
```

### Event Extraction

Event Extraction (EE) refers to extracting predefined Trigger and Argument from natural language texts and combining them into structured event information.

For example, if the targets are"地震强度", "时间", "震中位置" and "引源深度" for the event "地震", we can execute the following codes.

```python
>>> schema = {"地震触发词": ["地震强度", "时间", "震中位置", "震源深度"]}
>>> uie.set_schema(schema)
>>> results = uie.predict(
    [
        "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，"
        "震源深度10千米。"
    ],
    return_dict=True)
>>> pprint(results)

# An output example
# [{'地震触发词': {'end': 58,
#             'probability': 0.9977425932884216,
#             'relation': {'地震强度': [{'end': 56,
#                                    'probability': 0.9980800747871399,
#                                    'start': 52,
#                                    'text': '3.5级'}],
#                          '时间': [{'end': 22,
#                                  'probability': 0.9853301644325256,
#                                  'start': 11,
#                                  'text': '5月16日06时08分'}],
#                          '震中位置': [{'end': 50,
#                                    'probability': 0.7874020934104919,
#                                    'start': 23,
#                                    'text': '云南临沧市凤庆县(北纬24.34度，东经99.98度)'}],
#                          '震源深度': [{'end': 67,
#                                    'probability': 0.9937973618507385,
#                                    'start': 63,
#                                    'text': '10千米'}]},
#             'start': 56,
#             'text': '地震'}}]
```

### Opinion Extraction

opinion extraction refers to the extraction of evaluation dimensions and opinions contained in the text.

For example, if the extraction target is the evaluation dimensions and their corresponding opinions and sentiment tendencies. We can execute the following codes：

```python
>>> schema = {"评价维度": ["观点词", "情感倾向[正向，负向]"]}
>>> uie.set_schema(schema)
>>> results = uie.predict(
    ["店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"], return_dict=True)
>>> pprint(results)

# An output example
# [{'评价维度': {'end': 20,
#            'probability': 0.9817039966583252,
#            'relation': {'情感倾向[正向，负向]': [{'end': 0,
#                                          'probability': 0.9966142177581787,
#                                          'start': 0,
#                                          'text': '正向'}],
#                         '观点词': [{'end': 22,
#                                  'probability': 0.9573966264724731,
#                                  'start': 21,
#                                  'text': '高'}]},
#            'start': 17,
#            'text': '性价比'}}]
```

### Sentiment Classification

Sentence-level sentiment classification, i.e., determining a sentence has a "positive" sentiment or "negative" sentiment. We can execute the following codes:

```python
>>> schema = ["情感倾向[正向，负向]"]
>>> uie.set_schema(schema)
>>> results = uie.predict(["这个产品用起来真的很流畅，我非常喜欢"], return_dict=True)
>>> pprint(results)

# An output example
# [{'情感倾向[正向，负向]': {'end': 0,
#                   'probability': 0.9990023970603943,
#                   'start': 0,
#                   'text': '正向'}}]
```

### Cross-task Extraction

For example, in a legal scenario where both entity extraction and relation extraction need to be performed. We can execute the following codes.

```python
>>> schema = ["法院", {"原告": "委托代理人"}, {"被告": "委托代理人"}]
>>> uie.set_schema(schema)
>>> results = uie.predict(
    [
        "北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师"
        "事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"
    ],
    return_dict=True)
>>> pprint(results)
# An output example
# [{'原告': {'end': 37,
#          'probability': 0.9949813485145569,
#          'relation': {'委托代理人': [{'end': 46,
#                                  'probability': 0.7956855297088623,
#                                  'start': 44,
#                                  'text': '李四'}]},
#          'start': 35,
#          'text': '张三'},
#   '法院': {'end': 10,
#          'probability': 0.9221072793006897,
#          'start': 0,
#          'text': '北京市海淀区人民法院'},
#   '被告': {'end': 67,
#          'probability': 0.8437348008155823,
#          'relation': {'委托代理人': [{'end': 92,
#                                  'probability': 0.7267124652862549,
#                                  'start': 90,
#                                  'text': '赵六'}]},
#          'start': 64,
#          'text': 'B公司'}}]
```

## UIEModel Python Interface

```python
fd.text.uie.UIEModel(model_file,
                     params_file,
                     vocab_file,
                     position_prob=0.5,
                     max_length=128,
                     schema=[],
                     runtime_option=None,
                     model_format=ModelFormat.PADDLE,
                     schema_language=SchemaLanguage.ZH)
```

UIEModel loading and initialization. Among them, `model_file`, `params_file` are Paddle inference documents exported by trained models. Please refer to [Model export](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2). `vocab_file`refers to the vocabulary file. The vocabulary of the UIE model UIE can be downloaded in [UIE configuration file](https://github.com/PaddlePaddle/PaddleNLP/blob/5401f01af85f1c73d8017c6b3476242fce1e6d52/model_zoo/uie/utils.py).

**Parameter**

> * **model_file**(str): Model file path
> * **params_file**(str): Parameter file path
> * **vocab_file**(str): Vocabulary file
> * **position_prob**(str): Position probability. The model will output positions with probability greater than `position_prob`, default is 0.5
> * **max_length**(int): Maximized length of input text. Input text subscript exceeding `max_length` will be truncated. Default is 128
> * **schema**(list|dict): Target information for extraction tasks
> * **runtime_option**(RuntimeOption): Backend inference configuration, the default is None, i.e., the default configuration
> * **model_format**(ModelFormat): Model format, and default is Paddle format
> * **schema_language**(SchemaLanguage): Schema language, and default is ZH（Chinese）. Currently supported language：ZH（Chinese），EN（English）

### set_schema Function

> ```python
> set_schema(schema)
> ```
> Set schema interface of the UIE model.
>
> **Parameter**
> > * **schema**(list|dict): Enter the data to be extracted from the text.
>
> **Return**
> Blank.

### predict Function

> ```python
> UIEModel.predict(texts, return_dict=False)
> ```
>
> Model prediction interface where input text list directly output extraction results.
>
> **Parameter**
>
> > * **texts**(list(str)): Enter the data to be extracted from the text.
> > * **return_dict**(bool): Whether to output UIE results in the form of dictionary, and default is False。
> **Return**
>
> > Return`dict(str, list(fastdeploy.text.C.UIEResult))`。

## Related Documents

[Details for UIE model](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md)

[How to export a UIE model](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)

[UIE C++ deployment](../cpp/README.md)
