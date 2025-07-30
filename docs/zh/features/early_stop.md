
# 早停功能

早停功能用于提前结束模型生成token的过程，具体来说早停功能会采取不同的策略，判断当前生成的token序列是否满足早停条件，如果满足则提前结束token生成。FastDeploy目前只支持repetition策略。

1. Repetition策略
   * Repetition策略通过检查生成高概率token的次数决定是否需要触发早停功能。
   * 具体来说，当某个batch生成token的概率连续超过用户设置的概率阈值达到用户指定的次数，将提前结束该batch的token生成过程。

## 使用说明

在启动服务时，添加早停功能的启动项。

* 在线推理启动示例：
  * 使用默认超参数：--enable-early-stop
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
  * 使用自定义超参数：--early-stop-config
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
* 离线推理示例
  * 使用默认超参数：enable_early_stop
    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-0.3B-Paddle"

    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=1, enable_early_stop=True)
    output = llm.generate(prompts="who are you?", use_tqdm=True, sampling_params=sampling_params)

    print(output)
    ```
  * 使用自定义超参数：early_stop_config
    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-0.3B-Paddle"
    early_stop_config = {"enable_early_stop":True, "window_size":1000, "threshold":0.9}
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=1, early_stop_config=early_stop_config)
    output = llm.generate(prompts="who are you?", use_tqdm=True, sampling_params=sampling_params)

    print(output)
    ```

## 参数说明

* `enable_early_stop`: (bool) 是否启用早停功能，默认设置为False。
* `strategy`: (str) 早停功能使用的策略，目前仅支持repetition策略，默认设置为"repetition"。
* `window_size`: (int) repetition策略中连续出现高概率token的次数上限，超过该次数将触发早停功能，默认设置为3000。
* `threshold`: (float) repetition策略中的高概率阈值，默认设置为0.99。
