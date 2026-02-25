import unittest
from collections import OrderedDict
from dataclasses import asdict  # Import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch  # Import MagicMock

import numpy as np
import paddle

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ParallelConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
)
from fastdeploy.engine import common_engine as common_engine_module
from fastdeploy.engine import engine as engine_module
from fastdeploy.engine.args_utils import EngineArgs  # Import EngineArgs
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.input.text_processor import DataProcessor as TextDataProcessor
from fastdeploy.input.v1.text_processor import DataProcessor as V1TextDataProcessor
from fastdeploy.model_executor.logits_processor import ThinkingBudgetLogitsProcessor
from fastdeploy.scheduler import SchedulerConfig

# Constants for dummy tokenizer (matches ThinkingBudgetLogitsProcessor)
THINKING_START_TOKEN_ID = 151667
THINKING_END_TOKEN_ID = 151668
NEW_LINE_TOKEN_ID = 198
VOCAB_SIZE = 151669  # Just slightly larger than our special tokens


class MockRequest:
    def __init__(self, req_id, prompt_ids, sampling_params):
        self.req_id = req_id
        self.prompt_ids = prompt_ids
        self.sampling_params = sampling_params
        self.pre_ids = []  # To simulate generated tokens


class MockModelRunner:
    """A minimal mock for GPUModelRunner/OfflineEngine to provide share_inputs."""

    def __init__(self, fd_config, max_num_seqs=4):
        self.fd_config = fd_config
        self.max_num_seqs = max_num_seqs
        self.max_model_len = 16
        self.share_inputs = {
            "stop_flags": paddle.full([max_num_seqs, 1], False, dtype="bool"),
            "req_ids": [f"req_{i}" for i in range(max_num_seqs)],
            "logits_processors_args": [{} for _ in range(max_num_seqs)],
            "prompt_ids": paddle.to_tensor(np.zeros((max_num_seqs, 10), dtype=np.int64)),  # Max prompt len 10
            "prompt_lens": paddle.to_tensor(np.zeros((max_num_seqs, 1), dtype=np.int64)),
            "pre_ids": paddle.to_tensor(np.full((max_num_seqs, self.max_model_len), -1, dtype=np.int64)),
            "step_idx": paddle.to_tensor(np.zeros((max_num_seqs, 1), dtype=np.int64)),
            "next_tokens": paddle.to_tensor(np.full((max_num_seqs, 1), -1, dtype=np.int64)),
        }
        self.sampling_metadata = type(
            "SamplingMetadata", (object,), {"logits_processors": [ThinkingBudgetLogitsProcessor(fd_config)]}
        )()

    def update_request_state(self, slot_id, req: MockRequest, pre_id=None, set_next_token=True):
        # Extend prompt_ids tensor if needed
        current_prompt_len = self.share_inputs["prompt_ids"].shape[1]
        if len(req.prompt_ids) > current_prompt_len:
            new_prompt_ids_tensor = paddle.zeros((self.max_num_seqs, len(req.prompt_ids)), dtype=paddle.int64)
            new_prompt_ids_tensor[:, :current_prompt_len] = self.share_inputs["prompt_ids"]
            self.share_inputs["prompt_ids"] = new_prompt_ids_tensor

        self.share_inputs["req_ids"][slot_id] = req.req_id
        self.share_inputs["prompt_ids"][slot_id, : len(req.prompt_ids)] = paddle.to_tensor(
            req.prompt_ids, dtype=paddle.int64
        )
        self.share_inputs["prompt_lens"][slot_id] = paddle.to_tensor(len(req.prompt_ids), dtype=paddle.int64)
        if req.sampling_params.logits_processors_args:
            self.share_inputs["logits_processors_args"][slot_id] = req.sampling_params.logits_processors_args
        if pre_id is not None:
            self.share_inputs["step_idx"][slot_id] = self.share_inputs["step_idx"][slot_id] + 1
            step_pos = int(self.share_inputs["step_idx"][slot_id].item()) - 1
            if 0 <= step_pos < self.share_inputs["pre_ids"].shape[1]:
                self.share_inputs["pre_ids"][slot_id, step_pos] = paddle.to_tensor(pre_id, dtype=paddle.int64)
            if set_next_token:
                self.share_inputs["next_tokens"][slot_id] = paddle.to_tensor(pre_id, dtype=paddle.int64)

    def generate_next_token(self, logits):
        """Simulates sampling the next token (here, just pick the highest logit)."""
        return paddle.argmax(logits, axis=-1).tolist()


class TestThinkingBudgetLogitsProcessor(unittest.TestCase):

    def setUp(self):
        # Mimic how FDConfig is built in other tests to ensure all sub-configs are present
        max_num_seqs = 4  # Default for MockModelRunner
        engine_args = EngineArgs(max_num_seqs=max_num_seqs)
        args_dict_from_engine_args = asdict(engine_args)  # Convert EngineArgs to dict for sub-configs

        self._fdconfig_patches = [
            patch.object(FDConfig, "read_from_config", return_value=None),
            patch.object(FDConfig, "postprocess", return_value=None),
            patch.object(FDConfig, "init_cache_info", return_value=None),
            patch.object(FDConfig, "check", return_value=None),
        ]
        for patcher in self._fdconfig_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        # Use MagicMock for ModelConfig to bypass its complex initialization and network calls
        mock_model_config = MagicMock()
        mock_model_config.dtype = "float32"  # ThinkingBudgetLogitsProcessor needs this as string for self.dtype
        mock_model_config.vocab_size = VOCAB_SIZE  # ThinkingBudgetLogitsProcessor needs this
        mock_model_config.paddle_dtype = paddle.float32  # For _get_initial_logits to create paddle tensors
        # Add any other attributes ModelConfig might be expected to have, if accessed.
        # Based on previous error, this should cover it.
        mock_model_config.max_model_len = 512
        mock_model_config.architectures = ["mock_arch"]  # Needed for some ModelConfig methods if they were called
        mock_model_config.enable_mm = False
        mock_model_config.model_format = "auto"
        mock_model_config.think_start_id = THINKING_START_TOKEN_ID
        mock_model_config.think_end_id = THINKING_END_TOKEN_ID
        mock_model_config.line_break_id = NEW_LINE_TOKEN_ID

        cache_config = CacheConfig(args_dict_from_engine_args)
        parallel_config = ParallelConfig(args_dict_from_engine_args)
        speculative_config = SpeculativeConfig(args_dict_from_engine_args)
        scheduler_config = SchedulerConfig(args_dict_from_engine_args)
        load_config = LoadConfig(args_dict_from_engine_args)
        graph_opt_config = GraphOptimizationConfig(args_dict_from_engine_args)
        structured_outputs_config = StructuredOutputsConfig(args_dict_from_engine_args)

        self.fd_config = FDConfig(
            model_config=mock_model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            speculative_config=speculative_config,
            scheduler_config=scheduler_config,
            load_config=load_config,
            graph_opt_config=graph_opt_config,
            structured_outputs_config=structured_outputs_config,
            router_config=MagicMock(),
            test_mode=True,
        )

    def _get_initial_logits(self, batch_size):
        """Returns dummy logits for a batch."""
        # Create logits where token 0 is always highest by default, so it keeps generating 0
        logits = paddle.full((batch_size, VOCAB_SIZE), -10.0, dtype=self.fd_config.model_config.paddle_dtype)
        # Make a specific token have a higher logit value to simulate choice
        logits[:, 0] = 0.0  # Default token, so it will generate 0
        return logits

    def test_thinking_budget_not_reached(self):
        # Scenario: Thinking budget is 5, but only 3 tokens are generated in thinking phase
        req_id = "test_req_1"
        prompt_ids = [1, 2, THINKING_START_TOKEN_ID, 3, 4, 5]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 5})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])  # last token from prompt

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        # Step 1: Simulate update_state after prompt processing
        processor.update_state(mock_runner.share_inputs)
        self.assertTrue(processor._states[req_id].started)
        self.assertFalse(processor._states[req_id].ended)
        self.assertEqual(processor._states[req_id].tokens_after_start, 3)  # (3, 4, 5) after THINKING_START

        # Step 2: Simulate one generation step (budget 5, generated 3 -> 4)
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)  # Update state before apply
        processed_logits = processor.apply(logits)

        # Logits should not be modified, it allows generating 0
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)
        # Note: MockRequest.pre_ids is not used by the processor, but MockModelRunner.pre_id is.
        # So we simulate the update to MockModelRunner's state directly for the next step.
        mock_runner.update_request_state(0, mock_req, pre_id=0)  # Update last token

        processor.update_state(mock_runner.share_inputs)  # Update state after generating token
        self.assertEqual(processor._states[req_id].tokens_after_start, 4)  # (3, 4, 5, 0)

        # Step 3: Simulate another generation step (budget 5, generated 4 -> 5)
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)  # Update state before apply
        processed_logits = processor.apply(logits)
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)
        mock_runner.update_request_state(0, mock_req, pre_id=0)  # Update last token

        processor.update_state(mock_runner.share_inputs)  # Update state after generating token
        self.assertEqual(processor._states[req_id].tokens_after_start, 5)  # (3, 4, 5, 0, 0)

        # LogitsProcessor should still not restrict as NEW_LINE_TOKEN is not yet last token

    def test_thinking_budget_reached_forces_newline(self):
        # Scenario: Budget is 3, after 3 tokens, it should force newline, then thinking_end
        req_id = "test_req_2"
        prompt_ids = [1, 2, THINKING_START_TOKEN_ID, 3]  # Initial 1 token after start
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 3})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        line_break_id = processor.line_break_token_id
        think_end_id = processor.think_end_token_id

        # Step 1: Initial state update (1 token after start)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].tokens_after_start, 1)
        self.assertFalse(processor._states[req_id].ended)
        self.assertEqual(processor._states[req_id].last_token_id, 3)

        # Step 2: Generate 2nd token (budget 3, generated 1 -> 2)
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, 0)  # Normal generation
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].tokens_after_start, 2)
        self.assertEqual(processor._states[req_id].last_token_id, 0)

        # Step 3: Generate 3rd token (budget 3, generated 2 -> 3). Next step should force NEW_LINE.
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, 0)  # Normal generation
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].tokens_after_start, 3)  # Budget is now met
        self.assertEqual(processor._states[req_id].last_token_id, 0)

        # Step 4: Budget reached, last token not NEW_LINE. Should force NEW_LINE_TOKEN_ID.
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)

        # Verify all other logits are -inf, only NEW_LINE_TOKEN_ID is 0.0
        # Use <= for comparison because paddle.full creates -10.0 and comparison can have precision issues
        other_logits = paddle.concat(
            [
                processed_logits[0, :line_break_id],
                processed_logits[0, line_break_id + 1 : VOCAB_SIZE],
            ],
            axis=0,
        )
        self.assertTrue(paddle.all(other_logits <= -10.0).item() or paddle.all(other_logits == -float("inf")).item())
        self.assertEqual(processed_logits[0, line_break_id].item(), 0.0)

        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, line_break_id)  # Forces NEW_LINE
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].tokens_after_start, 4)  # Still increment
        self.assertEqual(processor._states[req_id].last_token_id, line_break_id)

        # Step 5: Last token is NEW_LINE. Should force THINKING_END_TOKEN_ID.
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)

        # Verify all other logits are -inf, only THINKING_END_TOKEN_ID is 0.0
        other_logits = paddle.concat(
            [
                processed_logits[0, :think_end_id],
                processed_logits[0, think_end_id + 1 : VOCAB_SIZE],
            ],
            axis=0,
        )
        self.assertTrue(paddle.all(other_logits <= -10.0).item() or paddle.all(other_logits == -float("inf")).item())
        self.assertEqual(processed_logits[0, think_end_id].item(), 0.0)

        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, think_end_id)  # Forces THINKING_END
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)

        # State should be ended
        self.assertTrue(processor._states[req_id].started)
        self.assertTrue(processor._states[req_id].ended)

        # Step 6: Thinking ended, processor should not interfere
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)  # Normal generation again

    def test_thinking_budget_stop_sentence_forces_tokens(self):
        # Scenario: stop sentence should be forced when budget threshold is reached.
        req_id = "test_req_stop_sentence"
        prompt_ids = [THINKING_START_TOKEN_ID]
        stop_sentence_token_ids = [10, 11, 12]
        sampling_params = SamplingParams(
            logits_processors_args={
                "thinking_budget": 10,
                "think_stop_sentence_token_ids": stop_sentence_token_ids,
            }
        )
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req)

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        # Initialize state from prompt
        processor.update_state(mock_runner.share_inputs)

        # Budget threshold = 10 - 3 = 7. Advance 7 tokens.
        for i in range(7):
            mock_runner.update_request_state(0, mock_req, pre_id=100 + i)
            processor.update_state(mock_runner.share_inputs)

        logits = self._get_initial_logits(1)
        processed_logits = processor.apply(logits)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, stop_sentence_token_ids[0])

    def test_thinking_budget_no_stop_sentence_defaults(self):
        # Scenario: No stop sentence, budget reached should force newline.
        req_id = "test_req_no_stop_sentence"
        prompt_ids = [THINKING_START_TOKEN_ID, 42]  # 1 token after start
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 1})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        line_break_id = processor.line_break_token_id

        processor.update_state(mock_runner.share_inputs)
        logits = self._get_initial_logits(1)
        processed_logits = processor.apply(logits)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, line_break_id)

    def test_thinking_budget_uses_config_token_ids(self):
        # Scenario: Processor should use token ids from model config.
        self.fd_config.model_config.think_start_id = 123
        self.fd_config.model_config.think_end_id = 124
        self.fd_config.model_config.line_break_id = 125
        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        self.assertEqual(processor.think_start_token_id, 123)
        self.assertEqual(processor.think_end_token_id, 124)
        self.assertEqual(processor.line_break_token_id, 125)
        self.assertTrue(processor._enabled)

    def test_thinking_budget_disabled_when_token_ids_missing(self):
        # Scenario: Processor should be disabled when token ids are not configured.
        self.fd_config.model_config.think_start_id = -1
        self.fd_config.model_config.think_end_id = -1
        self.fd_config.model_config.line_break_id = -1
        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        self.assertFalse(processor._enabled)
        self.assertEqual(processor.think_start_token_id, -1)
        self.assertEqual(processor.think_end_token_id, -1)
        self.assertEqual(processor.line_break_token_id, -1)

        # update_state and apply should be no-op when disabled
        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(len(processor._active_req_ids), 0)

        logits = self._get_initial_logits(1)
        processed_logits = processor.apply(logits)
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)

    def test_thinking_budget_stop_sentence_budget_smaller_than_sentence(self):
        # Scenario: budget smaller than stop sentence length should force stop sentence immediately
        req_id = "test_req_stop_sentence_small_budget"
        prompt_ids = [THINKING_START_TOKEN_ID]
        stop_sentence_token_ids = [101, 102, 103]
        sampling_params = SamplingParams(
            logits_processors_args={
                "thinking_budget": 1,
                "think_stop_sentence_token_ids": stop_sentence_token_ids,
            }
        )
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1], set_next_token=False)

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        processor.update_state(mock_runner.share_inputs)

        # Force stop sentence tokens
        for expected in stop_sentence_token_ids:
            logits = self._get_initial_logits(1)
            processor.update_state(mock_runner.share_inputs)
            processed_logits = processor.apply(logits)
            next_token = mock_runner.generate_next_token(processed_logits)[0]
            self.assertEqual(next_token, expected)
            mock_runner.update_request_state(0, mock_req, pre_id=next_token)
            processor.update_state(mock_runner.share_inputs)

        # After stop sentence, should force THINKING_END_TOKEN_ID
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, THINKING_END_TOKEN_ID)

    def test_thinking_budget_start_generated_token(self):
        # Scenario: Prompt has no think start, but model generates it during decoding
        req_id = "test_req_generated_think_start"
        prompt_ids = [1, 2, 3]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 2})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1], set_next_token=False)

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        processor.update_state(mock_runner.share_inputs)
        self.assertFalse(processor._states[req_id].started)

        # Simulate generated <think> token
        mock_runner.update_request_state(0, mock_req, pre_id=THINKING_START_TOKEN_ID)
        processor.update_state(mock_runner.share_inputs)
        self.assertTrue(processor._states[req_id].started)
        self.assertEqual(processor._states[req_id].tokens_after_start, 0)

    def test_thinking_budget_pre_ids_fallback_out_of_range(self):
        # Scenario: next_tokens is -1, fallback to pre_ids last token when step_pos is out of range
        req_id = "test_req_pre_ids_fallback"
        prompt_ids = [THINKING_START_TOKEN_ID]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 3})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1], set_next_token=False)

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        processor.update_state(mock_runner.share_inputs)
        # Force step_idx beyond pre_ids length and ensure next_tokens is invalid
        mock_runner.share_inputs["step_idx"][0, 0] = paddle.to_tensor(100, dtype=paddle.int64)
        mock_runner.share_inputs["next_tokens"][0, 0] = paddle.to_tensor(-1, dtype=paddle.int64)
        mock_runner.share_inputs["pre_ids"][0, -1] = paddle.to_tensor(77, dtype=paddle.int64)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].last_token_id, 77)

    def test_thinking_budget_cleanup_inactive_requests(self):
        # Scenario: stop_flag becomes True, state should be cleaned up
        req_id = "test_req_cleanup"
        prompt_ids = [THINKING_START_TOKEN_ID, 9]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 2})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        processor.update_state(mock_runner.share_inputs)
        self.assertIn(req_id, processor._states)

        mock_runner.share_inputs["stop_flags"][0] = paddle.to_tensor(True)
        processor.update_state(mock_runner.share_inputs)
        self.assertNotIn(req_id, processor._states)

    def test_thinking_budget_prompt_state_from_args(self):
        # 场景：使用预计算的 prompt 状态，避免 GPU 侧扫描 prompt
        req_id = "req_0"
        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.share_inputs["req_ids"][0] = req_id
        mock_runner.share_inputs["logits_processors_args"][0] = {
            "thinking_budget": 3,
            "think_prompt_checked": True,
            "think_prompt_started": True,
            "think_prompt_ended": False,
            "think_prompt_tokens_after_start": 2,
            "think_prompt_last_token_id": 99,
        }
        mock_runner.share_inputs["prompt_ids"] = None
        mock_runner.share_inputs["prompt_lens"] = None
        mock_runner.share_inputs["next_tokens"][0, 0] = paddle.to_tensor(-1, dtype=paddle.int64)

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)
        processor.update_state(mock_runner.share_inputs)

        state = processor._states[req_id]
        self.assertTrue(state.prompt_checked)
        self.assertTrue(state.started)
        self.assertFalse(state.ended)
        self.assertEqual(state.tokens_after_start, 2)
        self.assertEqual(state.last_token_id, 99)

    def test_thinking_budget_not_configured(self):
        # Scenario: Processor is active, but request does not provide thinking_budget
        req_id = "test_req_3"
        prompt_ids = [1, 2, THINKING_START_TOKEN_ID, 3]
        sampling_params = SamplingParams(logits_processors_args={})  # No thinking_budget
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        processor.update_state(mock_runner.share_inputs)
        # Processor should not be active for this request if budget is None
        self.assertNotIn(req_id, processor._states)
        self.assertFalse(processor._active_req_ids)

        logits = self._get_initial_logits(1)
        processed_logits = processor.apply(logits)
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)  # Normal generation

    def test_thinking_budget_zero(self):
        # Scenario: Thinking budget is 0, should force THINKING_END immediately
        req_id = "test_req_4"
        prompt_ids = [1, 2, THINKING_START_TOKEN_ID]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 0})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        processor.update_state(mock_runner.share_inputs)
        self.assertTrue(processor._states[req_id].started)
        self.assertFalse(processor._states[req_id].ended)
        self.assertEqual(processor._states[req_id].tokens_after_start, 0)  # No tokens after start yet
        self.assertEqual(processor._states[req_id].last_token_id, THINKING_START_TOKEN_ID)

        # Step 1: Budget 0 reached, last token is THINKING_START. Should force NEW_LINE.
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)

        self.assertEqual(processed_logits[0, NEW_LINE_TOKEN_ID].item(), 0.0)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, NEW_LINE_TOKEN_ID)
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)
        self.assertEqual(processor._states[req_id].tokens_after_start, 1)  # Still increments
        self.assertEqual(processor._states[req_id].last_token_id, NEW_LINE_TOKEN_ID)

        # Step 2: Last token is NEW_LINE. Should force THINKING_END_TOKEN_ID.
        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)

        self.assertEqual(processed_logits[0, THINKING_END_TOKEN_ID].item(), 0.0)
        next_token = mock_runner.generate_next_token(processed_logits)[0]
        self.assertEqual(next_token, THINKING_END_TOKEN_ID)
        mock_runner.update_request_state(0, mock_req, pre_id=next_token)
        processor.update_state(mock_runner.share_inputs)
        self.assertTrue(processor._states[req_id].ended)

    def test_thinking_end_in_prompt(self):
        # Scenario: THINKING_START and THINKING_END are already in the prompt
        req_id = "test_req_5"
        prompt_ids = [1, THINKING_START_TOKEN_ID, 2, THINKING_END_TOKEN_ID, 3]
        sampling_params = SamplingParams(logits_processors_args={"thinking_budget": 5})
        mock_req = MockRequest(req_id, prompt_ids, sampling_params)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=1)
        mock_runner.update_request_state(0, mock_req, pre_id=prompt_ids[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        processor.update_state(mock_runner.share_inputs)
        self.assertTrue(processor._states[req_id].started)
        self.assertTrue(processor._states[req_id].ended)
        self.assertEqual(processor._states[req_id].tokens_after_start, 2)  # Tokens 2, THINKING_END after start

        logits = self._get_initial_logits(1)
        processor.update_state(mock_runner.share_inputs)
        processed_logits = processor.apply(logits)
        self.assertEqual(paddle.argmax(processed_logits, axis=-1).item(), 0)  # Normal generation

    def test_multiple_requests(self):
        # Scenario: Multiple requests with different thinking states
        req_id_1 = "req_a"
        prompt_ids_1 = [THINKING_START_TOKEN_ID, 10, 11]  # budget 2, last_token=11, tokens_after_start=2
        sampling_params_1 = SamplingParams(logits_processors_args={"thinking_budget": 2})
        mock_req_1 = MockRequest(req_id_1, prompt_ids_1, sampling_params_1)

        req_id_2 = "req_b"
        prompt_ids_2 = [20, 21]  # no thinking, no budget
        sampling_params_2 = SamplingParams()
        mock_req_2 = MockRequest(req_id_2, prompt_ids_2, sampling_params_2)

        req_id_3 = "req_c"
        prompt_ids_3 = [THINKING_START_TOKEN_ID, 30]  # budget 1, last_token=30, tokens_after_start=1
        sampling_params_3 = SamplingParams(logits_processors_args={"thinking_budget": 1})
        mock_req_3 = MockRequest(req_id_3, prompt_ids_3, sampling_params_3)

        mock_runner = MockModelRunner(self.fd_config, max_num_seqs=3)
        mock_runner.share_inputs["stop_flags"] = paddle.full([3, 1], False, dtype="bool")
        mock_runner.update_request_state(0, mock_req_1, pre_id=prompt_ids_1[-1])
        mock_runner.update_request_state(1, mock_req_2, pre_id=prompt_ids_2[-1])
        mock_runner.update_request_state(2, mock_req_3, pre_id=prompt_ids_3[-1])

        processor = ThinkingBudgetLogitsProcessor(self.fd_config)

        processor.update_state(mock_runner.share_inputs)

        # Verify initial states
        self.assertTrue(processor._states[req_id_1].started)
        self.assertFalse(processor._states[req_id_1].ended)
        self.assertEqual(processor._states[req_id_1].tokens_after_start, 2)
        self.assertEqual(processor._states[req_id_1].last_token_id, 11)

        self.assertNotIn(req_id_2, processor._states)  # No budget specified for req_2

        self.assertTrue(processor._states[req_id_3].started)
        self.assertFalse(processor._states[req_id_3].ended)
        self.assertEqual(processor._states[req_id_3].tokens_after_start, 1)
        self.assertEqual(processor._states[req_id_3].last_token_id, 30)

        # Simulate logits for the batch
        batch_logits = self._get_initial_logits(3)
        processor.update_state(mock_runner.share_inputs)  # Ensure state is updated before apply
        processed_batch_logits = processor.apply(batch_logits)

        # Req 1: budget 2, tokens_after_start 2. Should force NEW_LINE (last_token_id is 11, not NEW_LINE)
        self.assertEqual(processed_batch_logits[0, NEW_LINE_TOKEN_ID].item(), 0.0)
        self.assertEqual(paddle.argmax(processed_batch_logits[0], axis=-1).item(), NEW_LINE_TOKEN_ID)

        # Req 2: No thinking budget, normal generation
        self.assertEqual(paddle.argmax(processed_batch_logits[1], axis=-1).item(), 0)

        # Req 3: budget 1, tokens_after_start 1. Should force NEW_LINE (last_token_id is 30, not NEW_LINE)
        self.assertEqual(processed_batch_logits[2, NEW_LINE_TOKEN_ID].item(), 0.0)
        self.assertEqual(paddle.argmax(processed_batch_logits[2], axis=-1).item(), NEW_LINE_TOKEN_ID)

        # Simulate generating next tokens and updating state
        next_tokens = mock_runner.generate_next_token(processed_batch_logits)
        mock_runner.update_request_state(0, mock_req_1, pre_id=next_tokens[0])
        mock_runner.update_request_state(1, mock_req_2, pre_id=next_tokens[1])
        mock_runner.update_request_state(2, mock_req_3, pre_id=next_tokens[2])
        processor.update_state(mock_runner.share_inputs)

        # Verify updated states for next step
        self.assertEqual(processor._states[req_id_1].last_token_id, NEW_LINE_TOKEN_ID)
        self.assertEqual(processor._states[req_id_3].last_token_id, NEW_LINE_TOKEN_ID)

        batch_logits = self._get_initial_logits(3)
        processor.update_state(mock_runner.share_inputs)
        processed_batch_logits = processor.apply(batch_logits)

        # Req 1: last token was NEW_LINE. Should force THINKING_END
        self.assertEqual(paddle.argmax(processed_batch_logits[0], axis=-1).item(), THINKING_END_TOKEN_ID)

        # Req 2: Still normal generation
        self.assertEqual(paddle.argmax(processed_batch_logits[1], axis=-1).item(), 0)

        # Req 3: last token was NEW_LINE. Should force THINKING_END
        self.assertEqual(paddle.argmax(processed_batch_logits[2], axis=-1).item(), THINKING_END_TOKEN_ID)


class DummyTokenizerForTextProcessor:
    def __init__(self):
        self.vocab = {"x": 0}

    def get_vocab(self):
        return {
            "<think>": THINKING_START_TOKEN_ID,
            "</think>": THINKING_END_TOKEN_ID,
        }

    def encode(self, text, add_special_tokens=False):
        return {"input_ids": [23]}


class DummyCfgRaiseParallel:
    @property
    def parallel_config(self):
        raise RuntimeError("stop-after-line-break")

    ips = None


class DummyRequestV1(SimpleNamespace):
    def get(self, key, default=None):
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value
        if hasattr(self, "sampling_params") and hasattr(self.sampling_params, key):
            value = getattr(self.sampling_params, key)
            if value is not None:
                return value
        return default

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def set(self, key, value):
        if hasattr(self, "sampling_params") and hasattr(self.sampling_params, key):
            setattr(self.sampling_params, key, value)
        else:
            setattr(self, key, value)


class TestThinkingBudgetSupplemental(unittest.TestCase):
    def test_update_thinking_prompt_state_from_text_processor(self):
        processor = TextDataProcessor.__new__(TextDataProcessor)
        processor._think_token_ids = None
        processor.tokenizer = DummyTokenizerForTextProcessor()
        prompt_ids = [1, THINKING_START_TOKEN_ID, 2, THINKING_END_TOKEN_ID, 3]
        args = {"thinking_budget": 5}
        updated = processor._update_thinking_prompt_state(prompt_ids, args)
        self.assertTrue(updated["think_prompt_checked"])
        self.assertTrue(updated["think_prompt_started"])
        self.assertTrue(updated["think_prompt_ended"])
        self.assertEqual(updated["think_prompt_tokens_after_start"], 2)
        self.assertEqual(updated["think_prompt_last_token_id"], 3)

    def test_v1_process_request_missing_logits_processors_args(self):
        processor = V1TextDataProcessor.__new__(V1TextDataProcessor)
        processor.generation_config = SimpleNamespace(
            top_p=0.7,
            temperature=1.0,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        processor.eos_token_ids = [1]
        processor.update_stop_seq = lambda *args, **kwargs: None
        processor.update_bad_words = lambda bad_words, bad_words_token_ids: bad_words_token_ids
        processor.encode_with_cache = lambda *args, **kwargs: [1]
        processor._update_thinking_prompt_state = lambda prompt_token_ids, args: args
        processor.reasoning_parser = None
        request = DummyRequestV1(
            request_id="req",
            eos_token_ids=None,
            prompt_token_ids=[1],
            prompt=None,
            messages=None,
            chat_template_kwargs=None,
            sampling_params=SimpleNamespace(
                bad_words=None,
                bad_words_token_ids=None,
                max_tokens=1,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            ),
        )
        with patch("fastdeploy.input.v1.text_processor.process_stop_token_ids", lambda *args, **kwargs: None):
            processor.process_request(request, max_model_len=8)

    def test_engine_line_break_id_from_dict(self):
        tokenizer = DummyTokenizerForTextProcessor()
        data_processor = SimpleNamespace(tokenizer=tokenizer, eos_token_id_len=1, pad_token_id=0)
        dummy_engine = SimpleNamespace(
            data_processor=data_processor,
        )
        engine = SimpleNamespace(
            data_processor=data_processor,
            engine=dummy_engine,
            cfg=DummyCfgRaiseParallel(),
        )
        engine._setting_environ_variables = lambda: ""
        with self.assertRaises(RuntimeError):
            engine_module.LLMEngine._start_worker_service(engine)

    def test_common_engine_line_break_id_from_dict(self):
        tokenizer = DummyTokenizerForTextProcessor()
        data_processor = SimpleNamespace(tokenizer=tokenizer, eos_token_id_len=1, pad_token_id=0)
        engine = SimpleNamespace(
            data_processor=data_processor,
            cfg=DummyCfgRaiseParallel(),
            llm_logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
        engine._setting_environ_variables = lambda: ""
        with self.assertRaises(RuntimeError):
            common_engine_module.EngineService._start_worker_service(engine)

    def test_text_encode_with_cache_branches(self):
        processor = TextDataProcessor.__new__(TextDataProcessor)
        processor._tokenize_cache = OrderedDict()
        processor._tokenize_cache_capacity = 1
        call_counter = {"np": 0, "iter": 0}

        def _text2ids(text, max_model_len=None, add_special_tokens=False):
            if text == "np":
                call_counter["np"] += 1
                return np.array([11, 12], dtype=np.int64)
            call_counter["iter"] += 1
            return (v for v in [21, 22])

        processor.text2ids = _text2ids

        self.assertEqual(processor.encode_with_cache("np"), [11, 12])
        self.assertEqual(processor.encode_with_cache("np"), [11, 12])
        self.assertEqual(call_counter["np"], 1)
        self.assertEqual(processor.encode_with_cache("iter"), [21, 22])
        self.assertNotIn(("np", False), processor._tokenize_cache)

    def test_v1_encode_with_cache_branches(self):
        processor = V1TextDataProcessor.__new__(V1TextDataProcessor)
        processor._tokenize_cache = OrderedDict()
        processor._tokenize_cache_capacity = 1
        call_counter = {"np": 0, "iter": 0}

        def _text2ids(text, max_model_len=None, add_special_tokens=False):
            if text == "np":
                call_counter["np"] += 1
                return np.array([31, 32], dtype=np.int64)
            call_counter["iter"] += 1
            return (v for v in [41, 42])

        processor.text2ids = _text2ids

        self.assertEqual(processor.encode_with_cache("np"), [31, 32])
        self.assertEqual(processor.encode_with_cache("np"), [31, 32])
        self.assertEqual(call_counter["np"], 1)
        self.assertEqual(processor.encode_with_cache("iter"), [41, 42])
        self.assertNotIn(("np", False), processor._tokenize_cache)

    def test_text_update_thinking_prompt_state_branches(self):
        processor = TextDataProcessor.__new__(TextDataProcessor)
        processor._think_token_ids = None
        processor.tokenizer = DummyTokenizerForTextProcessor()

        self.assertEqual(processor._update_thinking_prompt_state([1], "not-dict"), "not-dict")
        self.assertEqual(
            processor._update_thinking_prompt_state([1], {"thinking_budget": -1}), {"thinking_budget": -1}
        )
        self.assertEqual(
            processor._update_thinking_prompt_state([1], {"thinking_budget": 1, "think_prompt_checked": True}),
            {"thinking_budget": 1, "think_prompt_checked": True},
        )
        self.assertEqual(processor._update_thinking_prompt_state(None, {"thinking_budget": 1}), {"thinking_budget": 1})
        self.assertEqual(processor._update_thinking_prompt_state([], {"thinking_budget": 1}), {"thinking_budget": 1})

        processor.tokenizer = SimpleNamespace(get_vocab=lambda: {})
        self.assertEqual(processor._update_thinking_prompt_state([1], {"thinking_budget": 1}), {"thinking_budget": 1})

        processor._think_token_ids = None
        processor.tokenizer = DummyTokenizerForTextProcessor()
        without_start = processor._update_thinking_prompt_state(
            [999, 998],
            {"thinking_budget": 1, "think_prompt_last_token_id": 777},
        )
        self.assertTrue(without_start["think_prompt_checked"])
        self.assertFalse(without_start["think_prompt_started"])
        self.assertNotIn("think_prompt_last_token_id", without_start)

        with_start_no_end = processor._update_thinking_prompt_state(
            np.array([1, THINKING_START_TOKEN_ID, 2, 3], dtype=np.int64),
            {"thinking_budget": 4},
        )
        self.assertTrue(with_start_no_end["think_prompt_started"])
        self.assertFalse(with_start_no_end["think_prompt_ended"])
        self.assertEqual(with_start_no_end["think_prompt_tokens_after_start"], 2)
        self.assertEqual(with_start_no_end["think_prompt_last_token_id"], 3)

        # 命中 _get_think_token_ids 的缓存分支
        self.assertEqual(processor._get_think_token_ids(), (THINKING_START_TOKEN_ID, THINKING_END_TOKEN_ID))

    def test_v1_update_thinking_prompt_state_branches(self):
        processor = V1TextDataProcessor.__new__(V1TextDataProcessor)
        processor._think_token_ids = None
        processor.tokenizer = DummyTokenizerForTextProcessor()

        self.assertEqual(processor._update_thinking_prompt_state([1], "not-dict"), "not-dict")
        self.assertEqual(
            processor._update_thinking_prompt_state([1], {"thinking_budget": -1}), {"thinking_budget": -1}
        )
        self.assertEqual(processor._update_thinking_prompt_state(None, {"thinking_budget": 1}), {"thinking_budget": 1})

        with_start_no_end = processor._update_thinking_prompt_state(
            np.array([1, THINKING_START_TOKEN_ID, 2, 3], dtype=np.int64),
            {"thinking_budget": 4},
        )
        self.assertTrue(with_start_no_end["think_prompt_started"])
        self.assertFalse(with_start_no_end["think_prompt_ended"])
        self.assertEqual(with_start_no_end["think_prompt_tokens_after_start"], 2)
        self.assertEqual(with_start_no_end["think_prompt_last_token_id"], 3)

        # 命中 _get_think_token_ids 的缓存分支
        self.assertEqual(processor._get_think_token_ids(), (THINKING_START_TOKEN_ID, THINKING_END_TOKEN_ID))

    def test_text_process_request_think_stop_sentence(self):
        processor = TextDataProcessor.__new__(TextDataProcessor)
        processor._apply_default_parameters = lambda request: request
        processor.eos_token_ids = [1]
        processor.update_stop_seq = lambda *args, **kwargs: None
        processor.update_bad_words = lambda bad_words, bad_words_token_ids: bad_words_token_ids
        processor.encode_with_cache = lambda text, *args, **kwargs: [23] if text == "\n" else [101, 102]
        processor._update_thinking_prompt_state = lambda prompt_token_ids, args: args
        processor.reasoning_parser = None

        request = DummyRequestV1(
            request_id="req_text",
            eos_token_ids=[1],
            prompt_token_ids=[8],
            prompt=None,
            messages=None,
            logits_processors_args={"thinking_budget": 20, "think_stop_sentence": "done"},
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=1,
            temperature=1.0,
            top_p=0.9,
        )
        with patch("fastdeploy.input.text_processor.process_stop_token_ids", lambda *args, **kwargs: None):
            processed = processor.process_request(request, max_model_len=16)
        self.assertEqual(
            processed.logits_processors_args.get("think_stop_sentence_token_ids"),
            [23, 101, 102],
        )
        self.assertNotIn("think_stop_sentence", processed.logits_processors_args)

    def test_text_process_request_dict_think_stop_sentence(self):
        processor = TextDataProcessor.__new__(TextDataProcessor)
        processor._apply_default_parameters = lambda request: request
        processor.eos_token_ids = [1]
        processor.update_stop_seq = lambda *args, **kwargs: None
        processor.update_bad_words = lambda bad_words, bad_words_token_ids: bad_words_token_ids
        processor.encode_with_cache = lambda text, *args, **kwargs: [23] if text == "\n" else [201, 202]
        processor._update_thinking_prompt_state = lambda prompt_token_ids, args: args
        processor.reasoning_parser = None

        request = {
            "request_id": "req_text_dict",
            "eos_token_ids": [1],
            "prompt_token_ids": [9],
            "prompt": None,
            "messages": None,
            "bad_words": None,
            "bad_words_token_ids": None,
            "logits_processors_args": {"thinking_budget": 20, "think_stop_sentence": "done"},
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 0.9,
        }
        with patch("fastdeploy.input.text_processor.process_stop_token_ids", lambda *args, **kwargs: None):
            processed = processor.process_request_dict(request, max_model_len=16)
        self.assertEqual(
            processed["logits_processors_args"].get("think_stop_sentence_token_ids"),
            [23, 201, 202],
        )
        self.assertNotIn("think_stop_sentence", processed["logits_processors_args"])

    def test_v1_process_request_think_stop_sentence(self):
        processor = V1TextDataProcessor.__new__(V1TextDataProcessor)
        processor._apply_default_parameters = lambda request: request
        processor.eos_token_ids = [1]
        processor.update_stop_seq = lambda *args, **kwargs: None
        processor.update_bad_words = lambda bad_words, bad_words_token_ids: bad_words_token_ids
        processor.encode_with_cache = lambda text, *args, **kwargs: [23] if text == "\n" else [301, 302]
        processor._update_thinking_prompt_state = lambda prompt_token_ids, args: args
        processor.reasoning_parser = None

        request = DummyRequestV1(
            request_id="req_v1",
            eos_token_ids=[1],
            prompt_token_ids=[10],
            prompt=None,
            messages=None,
            logits_processors_args={"thinking_budget": 20, "think_stop_sentence": "done"},
            bad_words=None,
            bad_words_token_ids=None,
            max_tokens=1,
            temperature=1.0,
            top_p=0.9,
        )
        with patch("fastdeploy.input.v1.text_processor.process_stop_token_ids", lambda *args, **kwargs: None):
            processed = processor.process_request(request, max_model_len=16)
        self.assertEqual(
            processed.logits_processors_args.get("think_stop_sentence_token_ids"),
            [23, 301, 302],
        )
        self.assertNotIn("think_stop_sentence", processed.logits_processors_args)

    def test_v1_process_request_dict_think_stop_sentence(self):
        processor = V1TextDataProcessor.__new__(V1TextDataProcessor)
        processor._apply_default_parameters = lambda request: request
        processor.eos_token_ids = [1]
        processor.update_stop_seq = lambda *args, **kwargs: None
        processor.update_bad_words = lambda bad_words, bad_words_token_ids: bad_words_token_ids
        processor.encode_with_cache = lambda text, *args, **kwargs: [23] if text == "\n" else [401, 402]
        processor._update_thinking_prompt_state = lambda prompt_token_ids, args: args
        processor.reasoning_parser = None

        request = DummyRequestV1(
            request_id="req_v1_dict",
            eos_token_ids=[1],
            prompt_token_ids=[11],
            prompt=None,
            messages=None,
            chat_template_kwargs=None,
            sampling_params=SimpleNamespace(
                bad_words=None,
                bad_words_token_ids=None,
                max_tokens=1,
                temperature=1.0,
                top_p=0.9,
                repetition_penalty=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logits_processors_args={"thinking_budget": 20, "think_stop_sentence": "done"},
            ),
        )
        with patch("fastdeploy.input.v1.text_processor.process_stop_token_ids", lambda *args, **kwargs: None):
            processed = processor.process_request_dict(request, max_model_len=16)
        self.assertEqual(
            processed.sampling_params.logits_processors_args.get("think_stop_sentence_token_ids"),
            [23, 401, 402],
        )
        self.assertNotIn("think_stop_sentence", processed.sampling_params.logits_processors_args)


if __name__ == "__main__":
    unittest.main()
