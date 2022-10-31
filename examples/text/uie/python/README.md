# 通用信息抽取 UIE Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成UIE模型在CPU/GPU，以及CPU上通过OpenVINO加速CPU端部署示例。执行如下脚本即可完成。

## 快速开始
```bash

#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/text/uie/python

# 下载UIE模型文件和词表，以uie-base模型为例
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz

# CPU推理
python infer.py --model_dir uie-base --device cpu
# GPU推理
python infer.py --model_dir uie-base --device gpu
# 使用OpenVINO推理
python infer.py --model_dir uie-base --device cpu --backend openvino --cpu_num_threads 8
```

运行完成后返回结果如下所示(仅截取NER任务的输出)。
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

## UIE模型各抽取任务使用方式

在UIE模型中，schema代表要抽取的结构化信息，所以UIE模型可通过设置不同的schema支持不同信息抽取任务。

### 初始化UIEModel

```python
import fastdeploy
from fastdeploy.text import UIEModel
model_dir = "uie-base"
model_path = os.path.join(model_dir, "inference.pdmodel")
param_path = os.path.join(model_dir, "inference.pdiparams")
vocab_path = os.path.join(model_dir, "vocab.txt")

runtime_option = fastdeploy.RuntimeOption()
schema = ["时间", "选手", "赛事名称"]

# 初始化UIE模型
uie = UIEModel(
    model_path,
    param_path,
    vocab_path,
    position_prob=0.5,
    max_length=128,
    schema=schema,
    runtime_option=runtime_option)
```

### 实体抽取

初始化阶段将schema设置为```["时间", "选手", "赛事名称"]```，可对输入的文本抽取时间、选手以及赛事名称三个信息。

```python
>>> from pprint import pprint
>>> results = uie.predict(
        ["2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"], return_dict=True)
>>> pprint(results)

# 示例输出
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

例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", 则可执行如下语句：

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

# 示例输出
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


### 关系抽取

关系抽取（Relation Extraction，简称RE），是指从文本中识别实体并抽取实体之间的语义关系，进而获取三元组信息，即<主体，谓语，客体>。

例如以"竞赛名称"作为抽取主体，抽取关系类型为"主办方"、"承办方"和"已举办次数", 则可执行如下语句：

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

# 示例输出
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

### 事件抽取

事件抽取 (Event Extraction, 简称EE)，是指从自然语言文本中抽取预定义的事件触发词(Trigger)和事件论元(Argument)，组合为相应的事件结构化信息。

例如抽取的目标是"地震"事件的"地震强度"、"时间"、"震中位置"和"震源深度"这些信息，则可执行如下代码：

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

# 示例输出
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

### 评论观点抽取

评论观点抽取，是指抽取文本中包含的评价维度、观点词。

例如抽取的目标是文本中包含的评价维度及其对应的观点词和情感倾向，可执行以下代码：

```python
>>> schema = {"评价维度": ["观点词", "情感倾向[正向，负向]"]}
>>> uie.set_schema(schema)
>>> results = uie.predict(
    ["店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"], return_dict=True)
>>> pprint(results)

# 示例输出
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

### 情感分类

句子级情感倾向分类，即判断句子的情感倾向是“正向”还是“负向”，可执行以下代码：

```python
>>> schema = ["情感倾向[正向，负向]"]
>>> uie.set_schema(schema)
>>> results = uie.predict(["这个产品用起来真的很流畅，我非常喜欢"], return_dict=True)
>>> pprint(results)

# 示例输出
# [{'情感倾向[正向，负向]': {'end': 0,
#                   'probability': 0.9990023970603943,
#                   'start': 0,
#                   'text': '正向'}}]
```

### 跨任务抽取

例如在法律场景同时对文本进行实体抽取和关系抽取，可执行以下代码：

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
# 示例输出
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

## UIEModel Python接口

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

UIEModel模型加载和初始化，其中`model_file`, `params_file`为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)，`vocab_file`为词表文件，UIE模型的词表可在[UIE配置文件](https://github.com/PaddlePaddle/PaddleNLP/blob/5401f01af85f1c73d8017c6b3476242fce1e6d52/model_zoo/uie/utils.py)中下载相应的UIE模型的vocab_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **vocab_file**(str): 词表文件
> * **position_prob**(str): 位置概率，模型将输出位置概率大于`position_prob`的位置，默认为0.5
> * **max_length**(int): 输入文本的最大长度。输入文本下标超过`max_length`的部分将被截断。默认为128
> * **schema**(list|dict): 抽取任务的目标信息。
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式
> * **schema_language**(SchemaLanguage): Schema语言。默认为ZH（中文），目前支持的语言种类包括：ZH（中文），EN（英文）。

### set_schema函数

> ```python
> set_schema(schema)
> ```
> 设置UIE模型的schema接口。
>
> **参数**
> > * **schema**(list|dict): 输入数据，待抽取文本列表。
>
> **返回**
> 空。

### predict函数

> ```python
> UIEModel.predict(texts, return_dict=False)
> ```
>
> 模型预测接口，输入文本列表直接输出抽取结果。
>
> **参数**
>
> > * **texts**(list(str)): 输入数据，待抽取文本列表。
> > * **return_dict**(bool): 是否以字典形式输出UIE结果，默认为False。
> **返回**
>
> > 返回`dict(str, list(fastdeploy.text.C.UIEResult))`, 详细可见[UIEResult说明](../../../../docs/api/text_results/uie_result.md)。

## 相关文档

[UIE模型详细介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md)

[UIE模型导出方法](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/README.md#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2)

[UIE C++部署方法](../cpp/README.md)
