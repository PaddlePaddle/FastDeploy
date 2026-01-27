import io
import os
import unittest
import urllib

from PIL import Image

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.input.ernie4_5_tokenizer import Ernie4_5Tokenizer

os.environ["MACA_VISIBLE_DEVICES"] = "0,1"
os.environ["FD_MOE_BACKEND"] = "cutlass"
os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
os.environ["FLAGS_weight_only_linear_arch"] = "80"
os.environ["FD_METAX_KVCACHE_MEM"] = "8"
os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"


MATERIAL_PATH = "/data/material"
MODEL_PATH = "/data/models/PaddlePaddle/ERNIE-4.5-VL-28B-A3B-Thinking"


class TestErnie28BVL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()

        cls.llm = LLM(
            model=cls.model_path,
            tensor_parallel_size=2,
            engine_worker_queue_port=8899,
            max_model_len=32768,
            quantization="wint8",
            disable_custom_all_reduce=True,
            enable_prefix_caching=False,
            graph_optimization_config={"use_cudagraph": False, "graph_opt_level": 0},
            limit_mm_per_prompt={"image": 100},
            reasoning_parser="ernie-45-vl",
            load_choices="default_v1",
        )

        # cls.sampling_params = SamplingParams(top_p=0.95, max_tokens=32768, temperature=0.1)
        cls.sampling_params = SamplingParams(top_p=0.95, max_tokens=32768, temperature=0)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""

        material_path = MATERIAL_PATH
        cls.model_path = MODEL_PATH
        tokenizer = Ernie4_5Tokenizer.from_pretrained(cls.model_path)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Introduce yourself in detail"}]},  # text
            {  # large image
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{material_path}/ophelia.jpg"}},
                    {"type": "text", "text": "告诉我这幅画的名字以及它作者的生平简介。"},
                ],
            },
            {  # video
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{material_path}/football.mp4"}},
                    {"type": "text", "text": "Describe the video in detail."},
                ],
            },
        ]

        def process_content(content):
            images, videos = [], []
            for part in content:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith(("http://", "file://")):
                        url = f"file://{url}"
                    with urllib.request.urlopen(url) as response:
                        image_bytes = response.read()
                        img = Image.open(io.BytesIO(image_bytes))
                    images.append(img)
                elif part["type"] == "video_url":
                    url = part["video_url"]["url"]
                    if not url.startswith(("http://", "file://")):
                        url = f"file://{url}"
                    with urllib.request.urlopen(url) as response:
                        video_bytes = response.read()
                    videos.append({"video": video_bytes, "max_frames": 30})
            return images, videos

        prompts = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, list):
                continue
            prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
            images, videos = process_content(content)
            prompts.append({"prompt": prompt, "multimodal_data": {"image": images, "video": videos}})

        cls.text_prompt = prompts[0]
        cls.text_answer_keyword = ["large language model", "Baidu"]
        cls.text_answer_easy_keyword = ["multimodal", "ERNIE", "PaddlePaddle"]

        cls.image_prompt = prompts[1]
        cls.image_answer_keyword = ["自然", "艺术", "英国"]
        cls.image_answer_easy_keyword = ["奥菲莉亚", "夏洛特女郎", "约翰·威廉·沃特豪斯", "约翰·艾佛雷特·米莱"]

        cls.video_prompt = prompts[2]
        cls.video_answer_keyword = ["足球", "学生", "球门", "白色"]
        cls.video_answer_easy_keyword = ["老师", "教练", "蓝色", "黑色", "蓝白"]

    def test_text(self):
        outputs = self.llm.generate(prompts=self.text_prompt, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs.text
        print(f"Generated: {generated_text!r}")

        assert all(keyword in generated_text for keyword in self.text_answer_keyword)
        assert any(keyword in generated_text for keyword in self.text_answer_easy_keyword)

    def test_image(self):
        outputs = self.llm.generate(prompts=self.image_prompt, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs.text
        print(f"Generated: {generated_text!r}")

        assert all(keyword in generated_text for keyword in self.image_answer_keyword)
        assert any(keyword in generated_text for keyword in self.image_answer_easy_keyword)

    def test_video(self):
        outputs = self.llm.generate(prompts=self.video_prompt, sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs.text
        print(f"Generated: {generated_text!r}")

        assert all(keyword in generated_text for keyword in self.video_answer_keyword)
        assert any(keyword in generated_text for keyword in self.video_answer_easy_keyword)


if __name__ == "__main__":
    unittest.main()
