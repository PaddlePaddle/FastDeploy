[简体中文](../zh/features/sampling.md)

# Sampling Strategies

Sampling strategies are used to determine how to select the next token from the output probability distribution of a model. FastDeploy currently supports multiple sampling strategies including Top-p, Top-k_Top-p, and Min-p Sampling.

1. Top-p Sampling

   * Top-p sampling truncates the probability cumulative distribution, considering only the most likely token set that reaches a specified threshold p.
   * It dynamically selects the number of tokens considered, ensuring diversity in the results while avoiding unlikely tokens.

2. Top-k_Top-p Sampling

   * Initially performs top-k sampling, then normalizes within the top-k results, and finally performs top-p sampling.
   * By limiting the initial selection range (top-k) and then accumulating probabilities within it (top-p), it improves the quality and coherence of the generated text.

3. Min-p Sampling

   * Min-p sampling calculates `pivot=max_prob * min_p`, then retains only tokens with probabilities greater than the `pivot` (setting others to zero) for subsequent sampling.
   * It filters out tokens with relatively low probabilities, sampling only from high-probability tokens to improve generation quality.

## Usage Instructions

During deployment, you can choose the sampling algorithm by setting the environment variable `FD_SAMPLING_CLASS`. Available values are `base`, `base_non_truncated`, `air`, or `rejection`.

**Algorithms Supporting Only Top-p Sampling**

* `base` (default): Directly normalizes using the `top_p` value, favoring tokens with greater probabilities.
* `base_non_truncated`: Strictly follows the Top-p sampling logic, first selecting the smallest set that reaches the cumulative probability of `top_p`, then normalizing these selected elements.
* `air`: This algorithm is inspired by [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and supports Top-p sampling.

**Algorithms Supporting Top-p and Top-k_Top-p Sampling**

* `rejection`: This algorithm is inspired by [flashinfer](https://github.com/flashinfer-ai/flashinfer) and allows flexible settings for `top_k` and `top_p` parameters for Top-p or Top-k_Top-p sampling.

## Configuration Method

### Top-p Sampling

1. During deployment, set the environment variable to select the sampling algorithm, default is base:

```bash
export FD_SAMPLING_CLASS=rejection # base, base_non_truncated, or air
```
2. When sending a request, specify the following parameters:

* Example request with curl:

```bash

curl -X POST "http://0.0.0.0:9222/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "How old are you"}
  ],
  "top_p": 0.8
}'
```

* Example request with Python:

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
    ],
    stream=True,
    top_p=0.8
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

### Top-k_Top-p Sampling

1. During deployment, set the environment variable to select the rejection sampling algorithm:

```bash
export FD_SAMPLING_CLASS=rejection
```

2. When sending a request, specify the following parameters:

* Example request with curl:

```bash
curl -X POST "http://0.0.0.0:9222/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "How old are you"}
  ],
  "top_p": 0.8,
  "top_k": 20
}'
```

* Example request with Python:

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
    ],
    stream=True,
    top_p=0.8,
    extra_body={"top_k": 20, "min_p":0.1}
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

### Min-p Sampling

If you want to use min-p sampling before top-p or top-k_top-p sampling, specify the following parameters when sending a request:

* Example request with curl:

```bash
curl -X POST "http://0.0.0.0:9222/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "How old are you"}
  ],
  "min_p": 0.1,
  "top_p": 0.8,
  "top_k": 20
}'
```

* Example request with Python:

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
    ],
    stream=True,
    top_p=0.8,
    extra_body={"top_k": 20, "min_p":0.1}
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

With the above configurations, you can flexibly choose and use the appropriate sampling strategy according to the needs of specific generation tasks.

## Parameter Description

`top_p`: The probability cumulative distribution truncation threshold, considering only the most likely token set that reaches this threshold. It is a float type, with a range of [0.0, 1.0]. When top_p=1.0, all tokens are considered; when top_p=0.0, it degenerates into greedy search.

`top_k`: The number of tokens with the highest sampling probability, limiting the sampling range to the top k tokens. It is an int type, with a range of [0, vocab_size].

`min_p`: Low probability filtering threshold, considering only the token set with probability greater than or equal to (`max_prob*min_p`). It is a float type, with a range of [0.0, 1.0].

# Bad Words

Used to prevent the model from generating certain specific words during the inference process. Commonly applied in safety control, content filtering, and behavioral constraints of the model.

## Usage Instructions

Include the `bad_words` or `bad_words_token_ids` parameter in the request:

* Example request with curl:

```bash
curl -X POST "http://0.0.0.0:9222/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "How are you"}
  ],
  "bad_words": [" well", " Today"]
}'
```

Equal to

```bash
curl -X POST "http://0.0.0.0:9222/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "How are you"}
  ],
  "bad_words_token_ids": [1622, 25062]
}'
```

* Example request with Python:

```python
import openai
host = "0.0.0.0"
port = "9222"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
    ],
    extra_body={"bad_words": [" well", " Today"]},
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

Equal to

```python
import openai
host = "0.0.0.0"
port = "9222"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "user", "content": "Hello, how are you?"},
    ],
    extra_body={"bad_words_token_ids": [1622, 25062]},
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

## Parameter Description

`bad_words`: List of forbidden words. Type: list of str. Each word must be a single token.

`bad_words_token_ids`: List of forbidden token ids. Type: list of int.
