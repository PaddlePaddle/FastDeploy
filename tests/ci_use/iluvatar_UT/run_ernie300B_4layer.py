import functools
import sys
import threading

from fastdeploy import LLM, SamplingParams
from fastdeploy.utils import set_random_seed


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator


@timeout(60)
def offline_infer_check():
    set_random_seed(123)

    prompts = [
        "Hello, my name is",
    ]

    # 采样参数
    sampling_params = SamplingParams(temperature=0.8, top_p=0.00001, max_tokens=16)

    # 加载模型
    llm = LLM(
        model="/data1/fastdeploy/ERNIE_300B_4L",
        tensor_parallel_size=8,
        max_model_len=8192,
        quantization="wint8",
        block_size=16,
    )

    # 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
    outputs = llm.generate(prompts, sampling_params)

    assert outputs[0].outputs.token_ids == [
        23768,
        97000,
        47814,
        59335,
        68170,
        183,
        49080,
        94717,
        82966,
        99140,
        31615,
        51497,
        94851,
        60764,
        10889,
        2,
    ], f"{outputs[0].outputs.token_ids}"
    print("PASSED")


if __name__ == "__main__":
    try:
        result = offline_infer_check()
        sys.exit(0)
    except TimeoutError:
        sys.exit(124)
    except Exception:
        sys.exit(1)
