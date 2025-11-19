"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# --- Mocking Setup ---
# 优先模拟这些懒加载的模块，以便在未安装这些库的环境中进行测试。
mock_torch = MagicMock()
mock_llguidance = MagicMock()
mock_llguidance_hf = MagicMock()
mock_llguidance_torch = MagicMock()

mock_torch.__spec__ = MagicMock()
mock_torch.distributed = MagicMock()

sys.modules["torch"] = mock_torch
sys.modules["llguidance"] = mock_llguidance
sys.modules["llguidance.hf"] = mock_llguidance_hf
sys.modules["llguidance.torch"] = mock_llguidance_torch

# 模拟设置完成后，再导入需要测试的模块
from fastdeploy.model_executor.guided_decoding.guidance_backend import (
    LLGuidanceProcessor,
)


def MockFDConfig():
    """创建一个用于测试的FDConfig模拟对象"""
    config = MagicMock()
    # --- 修复点 1: 显式设置 model 为字符串，通过 HF 的验证 ---
    config.model_config.model = "test-model-path"
    config.model_config.architectures = []  # 设置为空列表，防止迭代 Mock 出错

    config.model_config.vocab_size = 1000
    config.scheduler_config.max_num_seqs = 4
    config.structured_outputs_config.disable_any_whitespace = False
    # 确保 backend 检查逻辑能通过
    config.structured_outputs_config.guided_decoding_backend = "guidance"
    return config


def MockHFTokenizer():
    """创建一个用于测试的Hugging Face Tokenizer模拟对象"""
    return MagicMock()


class TestLLGuidanceProcessorMocked(unittest.TestCase):
    """
    使用Mock对LLGuidanceProcessor进行单元测试。
    这个测试类适用于没有安装llguidance库的环境。
    """

    def setUp(self):
        """为每个测试用例设置一个新的LLGuidanceProcessor实例"""
        self.mock_matcher = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.eos_token = 2  # 示例EOS token ID
        self.processor = LLGuidanceProcessor(
            ll_matcher=self.mock_matcher,
            ll_tokenizer=self.mock_tokenizer,
            serialized_grammar="test_grammar",
            vocab_size=1000,
            batch_size=4,
            enable_thinking=False,
        )

    def test_init(self):
        """测试LLGuidanceProcessor的构造函数"""
        self.assertIs(self.processor.matcher, self.mock_matcher)
        self.assertEqual(self.processor.vocab_size, 1000)
        self.assertEqual(self.processor.batch_size, 4)
        self.assertFalse(self.processor.is_terminated)

    @patch("fastdeploy.utils.llm_logger.warning")
    def test_check_error_logs_warning_once(self, mock_log_warning):
        """测试_check_error方法在匹配器出错时能记录警告，且只记录一次"""
        self.mock_matcher.get_error.return_value = "A test error."

        # 第一次调用，应该打印日志
        self.processor._check_error()
        mock_log_warning.assert_called_once_with("LLGuidance Matcher error: A test error.")

        # 第二次调用，不应该重复打印
        self.processor._check_error()
        mock_log_warning.assert_called_once()

    @patch("fastdeploy.model_executor.guided_decoding.guidance_backend.llguidance_torch")
    def test_allocate_token_bitmask(self, mock_backend_torch):
        """
        测试token bitmask的分配。
        注意：这里Patch的是guidance_backend模块中导入的llguidance_torch变量，
        而不是sys.modules里的全局mock，以解决LazyLoader导致的引用不一致问题。
        """
        mock_backend_torch.allocate_token_bitmask.return_value = "fake_bitmask_tensor"

        result = self.processor.allocate_token_bitmask()

        mock_backend_torch.allocate_token_bitmask.assert_called_once_with(4, 1000)
        self.assertEqual(result, "fake_bitmask_tensor")

    @patch("fastdeploy.model_executor.guided_decoding.guidance_backend.llguidance_torch")
    def test_fill_token_bitmask(self, mock_backend_torch):
        """测试token bitmask的填充"""
        mock_bitmask = MagicMock()

        self.processor.fill_token_bitmask(mock_bitmask, idx=2)

        mock_backend_torch.fill_next_token_bitmask.assert_called_once_with(self.mock_matcher, mock_bitmask, 2)
        self.mock_matcher.get_error.assert_called_once()

    def test_reset(self):
        """测试处理器的状态重置"""
        self.processor.is_terminated = True
        self.processor._printed_error = True
        self.mock_matcher.get_error.return_value = ""

        self.processor.reset()

        self.mock_matcher.reset.assert_called_once()
        self.assertFalse(self.processor.is_terminated)
        self.assertFalse(self.processor._printed_error)

    def test_accept_token_when_terminated(self):
        """测试当状态为is_terminated时，accept_token直接返回False"""
        self.processor.is_terminated = True
        self.assertFalse(self.processor.accept_token(123))

    def test_accept_token_when_matcher_stopped(self):
        """测试当匹配器停止时，accept_token返回False并更新状态"""
        self.mock_matcher.is_stopped.return_value = True
        self.assertFalse(self.processor.accept_token(123))
        self.assertTrue(self.processor.is_terminated)

    def test_accept_token_is_eos(self):
        """测试接收到EOS token时的行为"""
        self.mock_matcher.is_stopped.return_value = False
        self.assertTrue(self.processor.accept_token(self.mock_tokenizer.eos_token))
        self.assertTrue(self.processor.is_terminated)

    def test_accept_token_consumes_and_succeeds(self):
        """测试成功消费一个token"""
        self.mock_matcher.is_stopped.return_value = False
        self.mock_matcher.consume_tokens.return_value = True
        self.assertTrue(self.processor.accept_token(123))
        self.mock_matcher.consume_tokens.assert_called_once_with([123])
        self.mock_matcher.get_error.assert_called_once()

    def test_accept_token_consumes_and_fails(self):
        """测试消费一个token失败"""
        self.mock_matcher.is_stopped.return_value = False
        self.mock_matcher.consume_tokens.return_value = False
        self.assertFalse(self.processor.accept_token(123))
        self.mock_matcher.consume_tokens.assert_called_once_with([123])


if __name__ == "__main__":
    unittest.main()
