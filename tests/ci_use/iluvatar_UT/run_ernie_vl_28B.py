import functools
import io
import sys
import threading

import requests
from PIL import Image

from fastdeploy import LLM, SamplingParams
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer
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


@timeout(180)
def offline_infer_check():
    set_random_seed(123)

    PATH = "/data1/fastdeploy/ERNIE-4.5-VL-28B-A3B-Paddle"
    tokenizer = Ernie4_5Tokenizer.from_pretrained(PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"
                    },
                },
                {"type": "text", "text": "图中的文物属于哪个年代"},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    images, videos = [], []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part["type"] == "image_url":
                url = part["image_url"]["url"]
                image_bytes = requests.get(url).content
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            elif part["type"] == "video_url":
                url = part["video_url"]["url"]
                video_bytes = requests.get(url).content
                videos.append({"video": video_bytes, "max_frames": 30})

    sampling_params = SamplingParams(temperature=0.1, max_tokens=16)
    llm = LLM(
        model=PATH,
        tensor_parallel_size=2,
        max_model_len=32768,
        block_size=16,
        quantization="wint8",
        limit_mm_per_prompt={"image": 100},
        reasoning_parser="ernie-45-vl",
    )
    outputs = llm.generate(
        prompts={"prompt": prompt, "multimodal_data": {"image": images, "video": videos}},
        sampling_params=sampling_params,
    )

    assert outputs[0].outputs.token_ids == [
        23,
        3843,
        94206,
        2075,
        52352,
        94133,
        13553,
        10878,
        93977,
        5119,
        93956,
        68725,
        14449,
        4356,
        38225,
        2,
    ], f"{outputs[0].outputs.token_ids}"
    print("PASSED")


if __name__ == "__main__":
    try:
        result = offline_infer_check()
        sys.exit(0)
    except TimeoutError:
        print(
            "The timeout exit may be due to multiple processes sharing the "
            "same gpu card. You can check this using ixsmi on the device."
        )
        sys.exit(124)
    except Exception:
        sys.exit(1)
