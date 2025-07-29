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

```json
{
  "top_p": 0.8
}
```

### Top-k_Top-p Sampling

1. During deployment, set the environment variable to select the rejection sampling algorithm:

```bash
export FD_SAMPLING_CLASS=rejection
```

2. When sending a request, specify the following parameters:

```json
{
  "top_p": 0.8,
  "top_k": 20
}
```

### Min-p Sampling

If you want to use min-p sampling before top-p or top-k_top-p sampling, specify the following parameters when sending a request:

```json
{
  "min_p": 0.1
}
```

With the above configurations, you can flexibly choose and use the appropriate sampling strategy according to the needs of specific generation tasks.

## Parameter Description

`top_p`: The probability cumulative distribution truncation threshold, considering only the most likely token set that reaches this threshold. It is a float type, with a range of [0.0, 1.0]. When top_p=1.0, all tokens are considered; when top_p=0.0, it degenerates into greedy search.

`top_k`: The number of tokens with the highest sampling probability, limiting the sampling range to the top k tokens. It is an int type, with a range of [0, vocab_size].

`min_p`: Low probability filtering threshold, considering only the token set with probability greater than or equal to (`max_prob*min_p`). It is a float type, with a range of [0.0, 1.0].

# Bad Words

Used to prevent the model from generating certain specific words during the inference process. Commonly applied in safety control, content filtering, and behavioral constraints of the model.

## Usage Instructions

Include the `bad_words` parameter in the request:

```json
{
  "bad_words": ["you", "me"]
}
```

## Parameter Description

`bad_words`: A list of strings, each representing a word or phrase that should not appear in the generated text.
