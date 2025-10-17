[简体中文](../zh/features/structured_outputs.md)

# Structured Outputs

## Overview

Structured Outputs refer to predefined format constraints that force large language models to generate content strictly following specified structures. This feature significantly improves output controllability and is suitable for scenarios requiring precise format outputs (such as API calls, data parsing, code generation, etc.), while supporting dynamic grammar extensions to balance flexibility and standardization.

FastDeploy supports using the [XGrammar](https://xgrammar.mlc.ai/docs/) backend to generate structured outputs.

Supported output formats:

- `json`: Output content in standard JSON format
- `regex`: Precise control over text patterns (e.g., dates, emails, ID numbers), supporting complex regex syntax
- `choice`: Output strictly limited to predefined options
- `grammar`: Uses extended BNF grammar to define complex structures, supporting field names, types and logical relationship constraints
- `structural tag`: Defines structured outputs through tag trees, supporting nested tags, attribute validation and mixed constraints

## Usage

When starting the service, you can specify the desired backend using the `--guided-decoding-backend` parameter. If set to `auto`, FastDeploy will automatically select the appropriate backend.

### OpenAI Interface

FastDeploy supports OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat Completions](https://platform.openai.com/docs/api-reference/chat) APIs. OpenAI specifies structured output formats through the `response_format` parameter.

FastDeploy ensures JSON output by specifying `json_object`. To define specific parameter names and types within JSON, use `json_schema`. Refer to examples for detailed usage.

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON object containing: names of China's Four Great Inventions, their dynasties of origin, and brief descriptions (each under 50 characters)",
        }
    ],
    response_format={"type": "json_object"}
)
print(completion.choices[0].message.content)
print("\n")
```

Output:

```json
{"inventions": [{"name": "Paper Making", "dynasty": "Han Dynasty", "description": "Invented during Han Dynasty, revolutionized writing and storage."}, {"name": "Printing", "dynasty": "Tang Dynasty", "description": "Woodblock printing developed in Tang, movable type in Song Dynasty."}, {"name": "Compass", "dynasty": "Han Dynasty (concept)", "description": "Early use in Han Dynasty, refined for navigation by Song Dynasty."}, {"name": "Gunpowder", "dynasty": "Tang Dynasty", "description": "Discovered in Tang Dynasty, later used in weapons and fireworks."}]}
```

The following example requires the model to return JSON data about a book, which must include `author`, `title`, and `genre` fields, with `genre` restricted to given `BookType` options.

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
            "content": "Generate a JSON describing a literary work, including author, title and book type.",
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

Output:

```json
{"author": "George Orwell", "title": "1984", "genre": "Dystopian"}
```

`structural_tag` can extract key information from input and call specified methods to return predefined structured output. The following example demonstrates getting a specified timezone using `structural_tag`:

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

content_str = """
You have the following function available:

{
    "name": "get_current_date",
    "description": "Get current date and time for given timezone",
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone to get current date/time, e.g.: Asia/Shanghai",
            }
        },
        "required": ["timezone"],
    }
}

If you choose to call only this function, reply in this format:
<{start_tag}={function_name}>{parameters}{end_tag}
where:

start_tag => `<function`
parameters => JSON dictionary with parameter names as keys
end_tag => `</function>`

Example:
<function=example_function>{"param": "value"}</function>

Note:
- Function call must follow specified format
- Required parameters must be specified
- Only one function can be called at a time
- Place entire function call response on a single line

You are an AI assistant. Answer the following question.
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
            "content": "You're traveling to Shanghai today",
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
                            "description": "Timezone to get current date/time, e.g.: Asia/Shanghai",
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

Output:

```
<function=get_current_date>{"timezone": "Asia/Shanghai"}</function>
```

For `choice`, `grammar`, and `regex` formats, FastDeploy supports them through the `extra_body` parameter. `choice` is straightforward - the model selects the best option from predefined choices. Example:

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "What is the landmark building in Shenzhen?"}
    ],
    extra_body={"guided_choice": ["Ping An Finance Centre", "China Resources Headquarters", "KK100", "Diwang Mansion"]},
)
print(completion.choices[0].message.content)
print("\n")
```

Output:

```
Ping An Finance Centre
```

`regex` allows restricting output format via regular expressions. Example for generating a standard web address:

```python
import openai

port = "8170"
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="null")

completion = client.chat.completions.create(
    model="null",
    messages=[
        {
            "role": "user",
            "content": "Generate a standard format web address including protocol and domain.\n",
        }
    ],
    extra_body={"guided_regex": r"^https:\/\/www\.[a-zA-Z]+\.com\/?$\n"},
)
print(completion.choices[0].message.content)
print("\n")
```

Output:

```
https://www.example.com/
```

`grammar` allows defining complex constraints using EBNF syntax. Example for generating HTML code:

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
            "content": "Generate HTML code for this heading in bold Times New Roman font: ERNIE Bot",
        }
    ],
    extra_body={"guided_grammar": html_h1_grammar},
)
print(completion.choices[0].message.content)
print("\n")
```

Output:

```
<h1 style="font-family: 'Times New Roman'; font-weight: bold">ERNIE Bot</h1>
```

### OpenAI Beta Interface

Starting from OpenAI Client v1.54.4, the `client.beta.chat.completions.parse` interface provides better support for Python native types. Example using `pydantic`:

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
        {"role": "user", "content": "The Oriental Pearl Tower is located at No.1 Century Avenue, Pudong New Area, Shanghai, with a height of 468 meters. What is its address and height?"},
    ],
    response_format=Info,
)

message = completion.choices[0].message
print(message)
print("\n")
assert message.parsed
print("Address:", message.parsed.addr)
print("Height:", message.parsed.height)
```

Output:

```
ParsedChatCompletionMessage[Info](content='{"addr": "No.1 Century Avenue, Pudong New Area, Shanghai", "height": 468}', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None, parsed=Info(addr='No.1 Century Avenue, Pudong New Area, Shanghai', height=468), reasoning_content=None)


Address: No.1 Century Avenue, Pudong New Area, Shanghai
Height: 468
```

### Offline Inference

Offline inference allows restricting the model's output format by pre-specified constraints. In `FastDeploy`, constraints can be specified through the `GuidedDecodingParams` class in `SamplingParams`. `GuidedDecodingParams` supports the following constraint types, with usage similar to online inference:

```python
json: Optional[Union[str, dict]] = None
regex: Optional[str] = None
choice: Optional[List[str]] = None
grammar: Optional[str] = None
json_object: Optional[bool] = None
structural_tag: Optional[str] = None
```

The following example demonstrates how to use offline inference to generate a structured json:

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
    prompts="Generate a JSON describing a literary work, including author, title and book type.",
    sampling_params=sampling_params,
)

# Output results
for output in outputs:
    print(output.outputs.text)
```

Output:

```
{"author": "George Orwell", "title": "1984", "genre": "Dystopian"}
```
