import os
import unittest

import fastdeploy

os.environ["MACA_VISIBLE_DEVICES"] = "0,1"
os.environ["FD_MOE_BACKEND"] = "cutlass"
os.environ["PADDLE_XCCL_BACKEND"] = "metax_gpu"
os.environ["FLAGS_weight_only_linear_arch"] = "80"
os.environ["FD_METAX_KVCACHE_MEM"] = "8"
os.environ["ENABLE_V1_KVCACHE_SCHEDULER"] = "1"
os.environ["FD_ENC_DEC_BLOCK_NUM"] = "2"


MODEL_PATH = "/data/models/PaddlePaddle/ERNIE-4.5-21B-A3B-Thinking"


class TestErnie21B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()

        cls.llm = fastdeploy.LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            engine_worker_queue_port=8899,
            max_model_len=256,
            quantization="wint8",
            load_choices="default_v1",
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            graph_optimization_config={"use_cudagraph": False, "graph_opt_level": 0},
        )

        cls.sampling_params = fastdeploy.SamplingParams(top_p=0.95, max_tokens=256, temperature=0.6)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""

        cls.text_prompt = [
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        ]

        cls.text_answer_keyword = ["fiber", "2 + 1 = 3"]

    def test_text(self):
        outputs = self.llm.generate(self.text_prompt, self.sampling_params)

        # prompt = outputs[0].prompt
        generated_text = outputs[0].outputs.text
        # print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")

        assert all(keyword in generated_text for keyword in self.text_answer_keyword)


if __name__ == "__main__":
    unittest.main()
