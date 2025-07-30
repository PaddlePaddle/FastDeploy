
# Early Stopping

The early stopping is used to prematurely terminate the token generation of the model. Specifically, the early stopping uses different strategies to determine whether the currently generated token sequence meets the early stopping criteria. If so, token generation is terminated prematurely. FastDeploy currently only supports the repetition strategy.

1. Repetition Strategy
* The repetition strategy determines whether to trigger the early stopping function by checking the number of times a high-probability token is generated.
* Specifically, if the probability of generating a token for a batch exceeds a user-set probability threshold for a specified number of consecutive times, token generation for that batch is terminated prematurely.

## Usage Instructions

When starting the service, add the early stopping function startup option.

* Online inference startup example:
  * Using default hyperparameters: --enable-early-stop
    ```shell
    python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-0.3B-Paddle \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --enable-early-stop
    ```
  * Using custom hyperparameters: --early-stop-config
    ```shell
    python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-0.3B-Paddle \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --early-stop-config '{"enable_early_stop":true, "window_size": 1000, "threshold": 0.9}'
    ```
* Offline reasoning example
  * Use default hyperparameter: enable_early_stop
    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-0.3B-Paddle"

    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=1, enable_early_stop=True)
    output = llm.generate(prompts="who are you?", use_tqdm=True, sampling_params=sampling_params)

    print(output)
    ```
  * Use custom hyperparameters: early_stop_config
    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-0.3B-Paddle"
    early_stop_config = {"enable_early_stop":True, "window_size":1000, "threshold":0.9}
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=1, early_stop_config=early_stop_config) output = llm.generate(prompts="who are you?", use_tqdm=True, sampling_params=sampling_params)

    print(output)
    ```

## Parameter Description

* `enable_early_stop`: (bool) Whether to enable the early stopping. Default False.

* `strategy`: (str) The strategy used by the early stopping. Currently, only the repetition strategy is supported. Default "repetition".

* `window_size`: (int) The upper limit of the number of consecutive high-probability tokens in the repetition strategy. If the number exceeds this limit, the early stopping will be triggered. Default 3000.

* `threshold`: (float) The high-probability threshold in the repetition strategy. Default 0.99.
