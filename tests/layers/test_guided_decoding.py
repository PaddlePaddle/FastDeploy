"""
测试GuidedDecoding类的单元测试
"""

import sys
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, Mock, patch

import paddle

mock_torch = MagicMock()
mock_xgrammar = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["xgrammar"] = mock_xgrammar

from fastdeploy.model_executor.guided_decoding import LogitsProcessorBase
from fastdeploy.model_executor.layers.sample.sampler import GuidedDecoding
from fastdeploy.reasoning import ReasoningParser


class TestGuidedDecoding(unittest.TestCase):
    """Test cases for GuidedDecoding class."""

    def setUp(self):
        """Setup for each test case."""
        # 创建一个基本的FDConfig对象
        self.fd_config = Mock()
        self.fd_config.scheduler_config = Mock()
        self.fd_config.scheduler_config.max_num_seqs = 5

        # 创建GuidedDecoding对象
        self.guided_decoding = GuidedDecoding(self.fd_config)

        # 创建一个模拟的LogitsProcessorBase
        self.mock_processor = Mock(spec=LogitsProcessorBase)
        self.mock_processor.is_terminated = False
        self.mock_processor.reasoning_ended = True
        self.mock_processor.enable_reasoning = False

        # 模拟allocate_token_bitmask方法返回一个假的bitmask
        self.mock_processor.allocate_token_bitmask.return_value = paddle.zeros([5, 10], dtype="int32")

        # 模拟fill_token_bitmask方法
        self.mock_processor.fill_token_bitmask.return_value = None

        # 模拟accept_token方法返回True
        self.mock_processor.accept_token.return_value = True

    def test_init(self):
        """Test initialization."""
        self.assertIsNone(self.guided_decoding.token_bitmask)
        self.assertEqual(len(self.guided_decoding.logits_processors), 5)
        self.assertIsNone(self.guided_decoding.reasoning_parser)
        self.assertEqual(len(self.guided_decoding._prefill_done_idxs), 5)
        self.assertEqual(len(self.guided_decoding._tokens_to_acc), 5)

    def test_apply_reasoning_parser(self):
        """Test apply_reasoning_parser method."""
        mock_parser = Mock(spec=ReasoningParser)
        self.guided_decoding.apply_reasoning_parser(mock_parser)
        self.assertEqual(self.guided_decoding.reasoning_parser, mock_parser)

    def test_add_logits_processor_no_future(self):
        """Test add_logits_processor method without future."""
        self.guided_decoding.add_logits_processor(0, None, [])
        self.assertFalse(self.guided_decoding._prefill_done_idxs[0])
        self.assertIsNone(self.guided_decoding.logits_processors[0])

    def test_add_logits_processor_with_prefill_tokens(self):
        """Test add_logits_processor method with prefill tokens."""
        # 创建模拟Future对象
        mock_future = Mock()
        mock_future.done.return_value = True
        mock_future.result.return_value = self.mock_processor

        prefill_tokens = [1, 2, 3]
        self.guided_decoding.add_logits_processor(0, mock_future, prefill_tokens)

        self.assertTrue(self.guided_decoding._prefill_done_idxs[0])
        self.assertEqual(self.guided_decoding.logits_processors[0], self.mock_processor)
        self.mock_processor.accept_token.assert_any_call(1)
        self.mock_processor.accept_token.assert_any_call(2)
        self.mock_processor.accept_token.assert_any_call(3)

    def test_add_logits_processor_with_async_future(self):
        """Test add_logits_processor method with async future."""
        # 创建模拟Future对象
        mock_future = Mock()
        mock_future.done.return_value = False

        prefill_tokens = [1, 2, 3]
        self.guided_decoding.add_logits_processor(0, mock_future, prefill_tokens)

        self.assertTrue(self.guided_decoding._prefill_done_idxs[0])
        self.assertEqual(self.guided_decoding.logits_processors[0], mock_future)
        self.assertEqual(self.guided_decoding._tokens_to_acc[0], prefill_tokens)

    def test_should_fill_bitmask_no_reasoning_parser(self):
        """Test should_fill_bitmask method with no reasoning parser."""
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.assertTrue(self.guided_decoding.should_fill_bitmask(0))

    def test_should_fill_bitmask_with_reasoning_parser(self):
        """Test should_fill_bitmask method with reasoning parser."""
        mock_parser = Mock(spec=ReasoningParser)
        self.guided_decoding.reasoning_parser = mock_parser

        # 测试 enable_reasoning=True 的情况
        self.mock_processor.enable_reasoning = True
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.assertTrue(self.guided_decoding.should_fill_bitmask(0))

        # 测试 enable_reasoning=False, reasoning_ended=False 的情况
        self.mock_processor.enable_reasoning = False
        self.mock_processor.reasoning_ended = False
        self.assertFalse(self.guided_decoding.should_fill_bitmask(0))

        # 测试 enable_reasoning=False, reasoning_ended=True 的情况
        self.mock_processor.reasoning_ended = True
        self.assertTrue(self.guided_decoding.should_fill_bitmask(0))

    def test_reset_processor(self):
        """Test reset_processor method."""
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding._prefill_done_idxs[0] = True

        self.guided_decoding.reset_processor(0)

        self.assertFalse(self.guided_decoding._prefill_done_idxs[0])
        self.assertIsNone(self.guided_decoding.logits_processors[0])

    def test_update_vocab_mask_with_new_prefill_done(self):
        """Test update_vocab_mask method with new prefill_done_idxs."""
        # 设置索引0的处理器
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding._prefill_done_idxs[0] = False

        # 调用update_vocab_mask并标记索引0为已完成
        self.guided_decoding.update_vocab_mask([0])

        # 验证_prefill_done_idxs[0]已更新
        self.assertTrue(self.guided_decoding._prefill_done_idxs[0])

        # 验证fill_token_bitmask被调用
        self.mock_processor.fill_token_bitmask.assert_called_once()

    def test_update_vocab_mask_with_future_processor(self):
        """Test update_vocab_mask method with future processor."""
        # 创建模拟Future对象
        mock_future = Mock()

        # 设置索引0的处理器为Future
        self.guided_decoding.logits_processors[0] = mock_future
        self.guided_decoding._prefill_done_idxs[0] = True

        # 调用update_vocab_mask
        self.guided_decoding.update_vocab_mask([])

        # 验证fill_token_bitmask没有被调用（因为处理器是Future）
        self.mock_processor.fill_token_bitmask.assert_not_called()

    def test_accept_tokens_from_prefill_node(self):
        """Test accept_tokens_from_prefill_node method."""
        # 设置索引0的处理器和待接受的tokens
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding._tokens_to_acc[0] = [1, 2, 3]

        # 调用accept_tokens_from_prefill_node
        self.guided_decoding.accept_tokens_from_prefill_node(0)

        # 验证accept_token被调用了3次
        self.assertEqual(self.mock_processor.accept_token.call_count, 3)
        self.mock_processor.accept_token.assert_any_call(1)
        self.mock_processor.accept_token.assert_any_call(2)
        self.mock_processor.accept_token.assert_any_call(3)

        # 验证_tokens_to_acc[0]已被重置
        self.assertIsNone(self.guided_decoding._tokens_to_acc[0])

    @patch("fastdeploy.model_executor.guided_decoding.xgrammar_backend.apply_token_mask")
    def test_apply_token_mask(self, mock_apply_token_mask):
        """Test apply_token_mask method."""
        # 创建测试数据
        logits = paddle.zeros([5, 10], dtype="float32")
        mock_apply_token_mask.return_value = paddle.ones([5, 10], dtype="float32")

        # 设置索引0的处理器
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding._prefill_done_idxs[0] = True

        # 调用apply_token_mask
        result = self.guided_decoding.apply_token_mask(logits, [])

        # 验证fill_token_bitmask没有被调用，非 Future
        self.mock_processor.fill_token_bitmask.assert_not_called()

        # 验证apply_token_mask被调用
        mock_apply_token_mask.assert_called_once()

        # 验证返回值
        self.assertTrue((result == paddle.ones([5, 10], dtype="float32")).all())

    def test_apply_token_mask_with_future_processor(self):
        """Test apply_token_mask method with future processor."""
        # 创建测试数据
        logits = paddle.zeros([5, 10], dtype="float32")

        # 创建模拟Future对象
        mock_future = Mock(spec=Future)
        mock_future.done.return_value = True
        mock_future.result.return_value = self.mock_processor

        # 设置索引0的处理器为Future
        self.guided_decoding.logits_processors[0] = mock_future

        self.guided_decoding._prefill_done_idxs[0] = True
        self.assertTrue(self.guided_decoding._prefill_done_idxs[0])
        self.assertIsNotNone(self.guided_decoding.logits_processors[0])
        self.assertTrue(isinstance(self.guided_decoding.logits_processors[0], Future))
        self.guided_decoding._tokens_to_acc[0] = [1, 2, 3]

        # 模拟patch apply_token_mask
        with patch(
            "fastdeploy.model_executor.guided_decoding.xgrammar_backend.apply_token_mask"
        ) as mock_apply_token_mask:
            mock_apply_token_mask.return_value = paddle.ones([5, 10], dtype="float32")

            # 调用apply_token_mask
            self.guided_decoding.apply_token_mask(logits, [])

        # 验证Future.result被调用
        mock_future.result.assert_called_once()

        # 验证accept_token被调用了3次
        self.assertEqual(self.mock_processor.accept_token.call_count, 3)

        # 验证_tokens_to_acc[0]已被重置
        self.assertIsNone(self.guided_decoding._tokens_to_acc[0])

    def test_accept_token(self):
        """Test _accept_token method."""
        # 设置索引0的处理器
        self.guided_decoding.logits_processors[0] = self.mock_processor

        # 调用_accept_token
        self.guided_decoding._accept_token(0, 1)

        # 验证accept_token被调用
        self.mock_processor.accept_token.assert_called_once_with(1)

    def test_accept_token_with_reasoning_parser(self):
        """Test _accept_token method with reasoning parser."""
        # 创建模拟ReasoningParser
        mock_parser = Mock(spec=ReasoningParser)
        mock_parser.is_reasoning_end.return_value = True
        self.guided_decoding.reasoning_parser = mock_parser

        # 设置索引0的处理器
        self.mock_processor.enable_reasoning = False
        self.mock_processor.reasoning_ended = False
        self.guided_decoding.logits_processors[0] = self.mock_processor

        # 调用_accept_token
        self.guided_decoding._accept_token(0, 1)

        # 验证is_reasoning_end被调用
        mock_parser.is_reasoning_end.assert_called_once_with([1])

        # 验证reasoning_ended已更新
        self.assertTrue(self.mock_processor.reasoning_ended)

        # 验证accept_token没有被调用（因为reasoning_ended刚被设置为True）
        self.mock_processor.accept_token.assert_not_called()

    def test_accept_token_processor_terminated(self):
        """Test _accept_token method when processor is terminated."""
        # 设置索引0的处理器，并让accept_token返回False
        self.mock_processor.accept_token.return_value = False
        self.guided_decoding.logits_processors[0] = self.mock_processor

        # 调用_accept_token
        self.guided_decoding._accept_token(0, 1)

        # 验证处理器被重置
        self.assertIsNone(self.guided_decoding.logits_processors[0])

    def test_update_output_tokens(self):
        """Test update_output_tokens method."""
        # 创建测试数据
        next_tokens = paddle.to_tensor([[1], [2], [3], [4], [5]])

        # 设置索引0和1的处理器
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding.logits_processors[1] = self.mock_processor
        self.guided_decoding._prefill_done_idxs[0] = True
        self.guided_decoding._prefill_done_idxs[1] = True

        # 调用update_output_tokens
        self.guided_decoding.update_output_tokens(next_tokens)

        # 验证accept_token被调用了两次
        self.assertEqual(self.mock_processor.accept_token.call_count, 2)
        self.mock_processor.accept_token.assert_any_call(1)
        self.mock_processor.accept_token.assert_any_call(2)

    def test_update_output_tokens_with_negative_token(self):
        """Test update_output_tokens method with negative token."""
        # 创建测试数据，包含负值
        next_tokens = paddle.to_tensor([[-1], [2]])

        # 设置索引0和1的处理器
        self.guided_decoding.logits_processors[0] = self.mock_processor
        self.guided_decoding.logits_processors[1] = self.mock_processor
        self.guided_decoding._prefill_done_idxs[0] = True
        self.guided_decoding._prefill_done_idxs[1] = True

        # 调用update_output_tokens
        self.guided_decoding.update_output_tokens(next_tokens)

        # 验证索引0的处理器被重置
        self.assertIsNone(self.guided_decoding.logits_processors[0])

        # 验证索引1的处理器的accept_token被调用
        self.mock_processor.accept_token.assert_called_once_with(2)

    def test_pre_process(self):
        """Test pre_process method."""
        # 模拟update_vocab_mask方法
        with patch.object(self.guided_decoding, "update_vocab_mask") as mock_update_vocab_mask:
            # 调用pre_process
            self.guided_decoding.pre_process([0, 1])

            # 验证update_vocab_mask被调用
            mock_update_vocab_mask.assert_called_once_with([0, 1])


if __name__ == "__main__":
    unittest.main()
