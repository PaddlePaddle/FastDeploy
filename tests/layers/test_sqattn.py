# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import unittest
import paddle

from fastdeploy.model_executor.ops.triton_ops import compress_model, process_model
from paddleformers.transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class DummyArgs:
    def __init__(self):
        self.model_name_or_path =  "Qwen/Qwen2.5-7B-Instruct"
        self.quant_bits = 4
        self.device = "cuda"
        self.max_eval_samples = 64
        self.max_new_tokens = 32
        self.seed = 123
        self.eval_batch_size = 1
        self.cache_dir = "./.cache"
        self.quant_method = "sqattn"
        self.output_dir = "./outputs/test_sqattn"
        self.temperature = 0.0
        self.do_sample = False
        self.qk_qtype = "int"
        self.v_qtype = "e4m3"
        self.eval_ppl = True
        self.quant = True
        self.bit8_thres = 0.75
        self.bit4_thres = 0.80
        self.plot_window_size_alloc = False
        self.use_relative_distance = True

class TestSQAttnPipeline(unittest.TestCase):

    def setUp(self):
        paddle.seed(123)
        self.args = DummyArgs()
        self.device = "cuda" if paddle.is_compiled_with_cuda() else "cpu"

    def test_full_compress_and_evaluate(self):
        
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)
        
        config = AutoConfig.from_pretrained(self.args.model_name_or_path, trust_remote_code=True)
        config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, use_fast=False, trust_remote_code=True
        )
        kwargs = {"dtype": paddle.bfloat16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path, config=config, **kwargs
        )
       
        print("[OK] model and tokenizer built.")
        
        sample_text = "The quick brown fox jumps over the lazy dog." * 100
        inputs = tokenizer(sample_text, return_tensors="pd")  
        model = process_model(model, None, self.args, method='naive')

        with paddle.no_grad():
            outputs = model(**inputs)
        
        bits_alloc = compress_model(model, tokenizer, self.device, self.args)
        model = process_model(model, bits_alloc, self.args, method='sqattn')
        self.assertIsNotNone(bits_alloc)

        model.eval()

        with paddle.no_grad():
            outputs = model(**inputs)
            if isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]

        self.assertIsNotNone(logits)
        self.assertFalse(paddle.isnan(logits).any().item())
        print(f"[OK] sample forward passed, logits shape: {list(logits.shape)}")

if __name__ == "__main__":
    if paddle.is_compiled_with_cuda():
        unittest.main()