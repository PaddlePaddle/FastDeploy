[English](../../features/structured_outputs.md)

# Structured Outputs

## 概述

Structured Outputs 是指通过预定义格式约束，使大模型生成内容严格遵循指定结构。该功能可显著提升生成结果的可控性，适用于需要精确格式输出的场景（如API调用、数据解析、代码生成等），同时支持动态语法扩展，平衡灵活性与规范性。

FastDeploy 支持使用 [XGrammar](https://xgrammar.mlc.ai/docs/) 后端生成结构化输出。

支持输出格式

- `json格式`: 输出内容为标准的 JSON 格式
- `正则表达式格式（regex）`: 精确控制文本模式（如日期、邮箱、身份证号等），支持复杂正则表达式语法
- `选项格式（choice）`: 输出严格限定在预定义选项中
- `语法格式（grammar）`: 使用扩展 BNF 语法定义复杂结构，支持字段名、类型及逻辑关系的联合约束
- `结构标签格式（structural_tag）`: 通过标签树定义结构化输出，支持嵌套标签、属性校验及混合约束

## 使用方式

服务启动时，可以通过 `--guided-decoding-backend` 参数指定期望使用的后端，如果指定 `auto`, FastDeploy 会自动选择合适的后端。

### OpenAI 接口

FastDeploy 支持 OpenAI 的 [Completions](https://platform.openai.com/docs/api-reference/completions) 和 [Chat Completions](https://platform.openai.com/docs/api-reference/chat) API，OpenAI 通过 `response_format` 参数指定 Structured Outputs 的输出格式。

FastDeploy 支持通过指定 `json_object` 来保证输出为json格式，如果想指定 JSON 内部具体参数名、类型等信息，可以通过 `json_schema` 来指定详细信息，详细使用方法可以参考示例。

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "生成一个包含以下信息的JSON对象：中国四大发明名称、发明朝代及简要描述（每项不超过50字）",
        }
    ],
    response_format={"type": "json_object"}
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
{"inventions": [{"name": "造纸术", "dynasty": "东汉", "description": "蔡伦改进造纸术，用树皮、麻头等为原料，成本低廉，推动文化传播。"}, {"name": "印刷术", "dynasty": "唐朝", "description": "雕版印刷术在唐朝出现，后毕昇发明活字印刷术，提高印刷效率。"}, {"name": "火药", "dynasty": "唐朝", "description": "古代炼丹家偶然发明，后用于军事，改变了战争方式和格局。"}, {"name": "指南针", "dynasty": "战国", "description": "最初称司南，后发展为指南针，为航海等提供方向指引。"}]}
```

以下示例要求模型返回一个 book 信息的 json 数据，json 数据中必须包含`author、title、genre`三个字段，且`genre`必须在给定 `BookType` 中。

```python
import openai
from pydantic import BaseModel
from enum import Enum

class BookType(str, Enum):
    romance = "Romance"
    historical = "Historical"
    adventure = "Adventure"
    mystery = "Mystery"
    dystopian = "Dystopian"

class BookDescription(BaseModel):
    author: str
    title: str
    genre: BookType

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "生成一个JSON，描述一本中国的著作，要包含作者、标题和书籍类型。",
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "book-description",
            "schema": BookDescription.model_json_schema()
        },
    },
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
{"author": "曹雪芹", "title": "红楼梦", "genre": "Historical"}
```

`structural_tag` 可以提取输入内容中的重点信息，并调用指定方法，返回预定义的结构化输出。以下示例展示了通过 `structural_tag` 获取指定时区并输出格式化结果。

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

content_str = """
你有以下函数可以调用：

{
    "name": "get_current_date",
    "description": "根据给定的时区获取当前日期和时间",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "获取当前日期和时间的时区, 例如: Asia/Shanghai",
            }
        },
        "required": ["timezone"],
    }
}

如果你选择只调用函数，请按照以下格式回复：
<{start_tag}={function_name}>{parameters}{end_tag}
其中

start_tag => `<function`
parameters => 一个JSON字典，其中函数参数名为键，函数参数值为值。
end_tag => `</function>`

这是一个示例，
<function=example_function_name>{"example_name": "example_value"}</function>

注意：
- 函数调用必须遵循指定的格式
- 必须指定所需参数
- 一次只能调用一个函数
- 将整个函数调用回复放在一行上

你是一个人工智能助理，请回答下列问题。
"""

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "system",
            "content": content_str,
        },
        {
            "role": "user",
            "content": "你今天去上海出差",
        }
    ],
    response_format={
        "type": "structural_tag",
        "structures": [
            {
                "begin": "<function=get_current_date>",
                "schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "获取当前日期和时间的时区, 例如: Asia/Shanghai",
                        }
                    },
                    "required": ["timezone"],
                },
                "end": "</function>",
            }
        ],
        "triggers": ["<function="],
    },
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
<function=get_current_date>{"timezone": "Asia/Shanghai"}</function>
```

对于 `choice、grammar、regex`格式，FastDeploy 使用 `extra_body` 参数支持，`choice` 较为简单，指定后模型会在预定义选项中选择一个最优项。在 FastDeploy 中，可以通过 `guided_choice` 指定一组与定义选项，示例如下

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "深圳的地标建筑是什么？"}
    ],
    extra_body={"guided_choice": ["深圳平安国际金融中心", "中国华润大厦（春笋大厦）", "深圳京基100", "深圳地王大厦"]},
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
深圳地王大厦
```

`regex` 允许用户通过正则表达式限制输出格式，通常用于需要精确控制文本格式的场景。以下示例展示了通过模型生成一个标准格式的网络地址的过程。

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "生成一个标准格式的网络地址，包括协议、域名。\n",
        }
    ],
    extra_body={"guided_regex": r"^https:\/\/www\.[a-zA-Z]+\.com\/?$\n"},
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
https://www.example.com/
```

`grammar` 允许用户通过 EBNF(Extended Backus-Naur Form) 语法描述一个限制规则，以下示例展示了通过 `guided_grammar` 限制模型输出一段html代码。

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

html_h1_grammar = """
    root ::= html_statement

    html_statement ::= "<h1" style_attribute? ">" text "</h1>"

    style_attribute ::= " style=" dq style_value dq

    style_value ::= (font_style ("; " font_weight)?) | (font_weight ("; " font_style)?)

    font_style ::= "font-family: '" font_name "'"

    font_weight ::= "font-weight: " weight_value

    font_name ::= "Arial" | "Times New Roman" | "Courier New"

    weight_value ::= "normal" | "bold"

    text ::= [A-Za-z0-9 ]+

    dq ::= ["]
"""

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "生成一段html的代码，对以下标题加粗、Times New Roman字体。标题：ERNIE Bot",
        }
    ],
    extra_body={"guided_grammar": html_h1_grammar},
)
print(completion.choices[0].message.content)
print("\n")
```

输出

```
<h1 style="font-family: 'Times New Roman'; font-weight: bold">ERNIE Bot</h1>
```

### OpenAI beta 接口

在 OpenAI Client 1.54.4 版本之后，提供了 `client.beta.chat.completions.parse` 接口，通过该接口，实现了与 Python 原生类型更好的支持。以下示例展示了使用该接口直接获取 `pydantic` 类型的结构化输出。

```python
from pydantic import BaseModel
import openai

class Info(BaseModel):
    addr: str
    height: int

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.beta.chat.completions.parse(
    model="null",
    messages=[
        {"role": "user", "content": "上海东方明珠在上海市浦东新区世纪大道1号，高度为468米，请问上海东方明珠的地址和高度是多少？"},
    ],
    response_format=Info,
)

message = completion.choices[0].message
print(message)
print("\n")
assert message.parsed
print("地址:", message.parsed.addr)
print("高度:", message.parsed.height)
```

输出

```
ParsedChatCompletionMessage[Info](content='{"addr": "上海市浦东新区世纪大道1号", "height": 468}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, parsed=Info(addr='上海市浦东新区世纪大道1号', height=468), reasoning_content=None)


地址: 上海市浦东新区世纪大道1号
高度: 468
```

### 离线推理

离线推理允许通过预先指定约束条件，限制模型输出格式。在 `FastDeploy` 中，支持通过 `SamplingParams` 中的 `GuidedDecodingParams` 类指定相关约束条件。`GuidedDecodingParams` 支持以下几种约束条件，使用方式可以参考在线推理：

```python
json: Optional[Union[str, dict]] = None
regex: Optional[str] = None
choice: Optional[List[str]] = None
grammar: Optional[str] = None
json_object: Optional[bool] = None
structural_tag: Optional[str] = None
```

以下示例展示了如何使用离线推理生成一个结构化的 json :

```python

from fastdeploy import LLM, SamplingParams
from fastdeploy.engine.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
from enum import Enum

class BookType(str, Enum):
    romance = "Romance"
    historical = "Historical"
    adventure = "Adventure"
    mystery = "Mystery"
    dystopian = "Dystopian"

class BookDescription(BaseModel):
    author: str
    title: str
    genre: BookType

# Constrained decoding parameters
guided_decoding_params = GuidedDecodingParams(json=BookDescription.model_json_schema())

# Sampling parameters
sampling_params = SamplingParams(
    top_p=0.95,
    max_tokens=6400,
    guided_decoding=guided_decoding_params,
)

# Load model
llm = LLM(model="ERNIE-4.5-0.3B", tensor_parallel_size=1, max_model_len=8192, guided_decoding_backend="auto")

outputs = llm.generate(
    prompts="生成一个JSON，描述一本中国的著作，要包含作者、标题和书籍类型。",
    sampling_params=sampling_params,
)

# Output results
for output in outputs:
    print(output.outputs.text)

```

输出

```
{"author": "曹雪芹", "title": "红楼梦", "genre": "Historical"}
```
