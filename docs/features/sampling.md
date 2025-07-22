# Sampling Strategies

Sampling strategies are used to decide how to select the next token from the model's output probability distribution. FastDeploy currently supports two sampling strategies: Top-p and Top-k_Top-p.

## Top-p Sampling

- **Top-p Sampling** truncates the probability cumulative distribution and only considers the most likely token set that reaches a specified threshold p.
- It dynamically selects the number of tokens considered, ensuring diversity in the results while avoiding less probable tokens.

## Top-k_Top-p Sampling

- **Top-k_Top-p Sampling** first performs top-k sampling, then normalizes the results of top-k, followed by top-p sampling.
- By limiting the initial selection range (top-k) and then performing cumulative probability selection (top-p) within this range, it enhances the quality and coherence of the generated text.

## Usage Instructions

During deployment, you can set the environment variable `FD_SAMPLING_CLASS` to choose the sampling algorithm. The available values are `base`, `base_non_truncated`, `air`, or `rejection`.

### Algorithms Supporting Only Top-p Sampling

- **`base` (default):** Directly normalizes using the `top_p` value, tending to sample tokens with higher probabilities.
- **`base_non_truncated`:** Strictly follows the logic of Top-p sampling, first selecting the minimum set that accumulates to the `top_p` probability, then normalizing these selected elements.
- **`air`:** This algorithm references the implementation from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and supports Top-p sampling.

### Algorithms Supporting Both Top-p and Top-k_Top-p Sampling

- **`rejection`:** This algorithm references the implementation from [flashinfer](https://github.com/flashinfer-ai/flashinfer) and allows flexible setting of `top_k` and `top_p` parameters for either Top-p or Top-k_Top-p sampling.

## Configuration Method

If you wish to use the top_k_top_p sampling strategy, follow these steps:

1. During deployment, set the environment variable to select the rejection sampling algorithm:

    ```bash
    export FD_SAMPLING_CLASS=rejection
    ```

2. When sending a request, specify the following parameters:

    ```json
    {
      "top_p": 0.8,
      "top_k": 20,
    }
    ```

    > If `top_k` is not specified in the request, Top-p sampling will be performed by default.

Through the above configuration, you can flexibly choose and use the appropriate sampling strategy according to the specific requirements of the generation task.

## Parameter Descriptions

- **top_p:** The probability cumulative distribution truncation threshold, only considers the most likely token set that reaches this threshold.
- **top_k:** The number of tokens with the highest sampling probability, limits the sampling range to the top k tokens with the highest probabilities.
