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

import time
import unittest

from fastdeploy.engine.request import (
    CompletionOutput,
    LogprobsLists,
    RequestMetrics,
    RequestOutput,
)


class TestRequestOutputInit(unittest.TestCase):
    """Test case for RequestOutput initialization"""

    def test_init_default_values(self):
        """Test initialization with default values"""
        request_id = "test_request_123"
        request_output = RequestOutput(request_id=request_id)

        self.assertEqual(request_output.request_id, request_id)
        self.assertIsNone(request_output.prompt)
        # prompt_token_ids becomes empty list when None is passed
        self.assertEqual(request_output.prompt_token_ids, [])
        self.assertIsNone(request_output.prompt_logprobs)
        self.assertEqual(request_output.output_type, 3)
        self.assertIsNone(request_output.outputs)
        self.assertFalse(request_output.finished)
        self.assertIsNone(request_output.metrics)
        self.assertEqual(request_output.num_cached_tokens, 0)
        self.assertEqual(request_output.num_input_image_tokens, 0)
        self.assertEqual(request_output.num_input_video_tokens, 0)
        self.assertEqual(request_output.error_code, 200)
        self.assertIsNone(request_output.error_msg)
        self.assertIsNone(request_output.ic_req_data)
        self.assertEqual(request_output.prompt_token_ids_len, 0)
        self.assertIsNone(request_output.accumulate_tool_calls)

    def test_init_with_numpy_array_prompt_token_ids(self):
        """Test initialization with numpy array prompt_token_ids"""
        import numpy as np

        request_id = "test_request_456"
        numpy_array = np.array([1, 2, 3, 4, 5])

        request_output = RequestOutput(request_id=request_id, prompt_token_ids=numpy_array)

        self.assertEqual(request_output.prompt_token_ids, [1, 2, 3, 4, 5])
        self.assertIsInstance(request_output.prompt_token_ids, list)

    def test_init_with_list_prompt_token_ids(self):
        """Test initialization with list prompt_token_ids"""
        request_id = "test_request_789"
        token_list = [10, 20, 30, 40]

        request_output = RequestOutput(request_id=request_id, prompt_token_ids=token_list)

        self.assertEqual(request_output.prompt_token_ids, token_list)

    def test_init_with_outputs_and_tool_calls(self):
        """Test initialization with outputs containing tool calls"""
        request_id = "test_request_tool"

        # Create a CompletionOutput with tool calls
        tool_calls = [{"name": "test_tool", "arguments": {"param": "value"}}]
        outputs = CompletionOutput(index=0, send_idx=0, token_ids=[100, 200, 300], tool_calls=tool_calls)

        request_output = RequestOutput(request_id=request_id, outputs=outputs)

        self.assertEqual(request_output.accumulate_tool_calls, [tool_calls])
        self.assertEqual(request_output.outputs, outputs)

    def test_init_with_outputs_no_tool_calls(self):
        """Test initialization with outputs but no tool calls"""
        request_id = "test_request_no_tool"

        # Create a CompletionOutput without tool calls
        outputs = CompletionOutput(index=0, send_idx=0, token_ids=[100, 200, 300])

        request_output = RequestOutput(request_id=request_id, outputs=outputs)

        self.assertIsNone(request_output.accumulate_tool_calls)
        self.assertEqual(request_output.outputs, outputs)

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided"""
        request_id = "test_request_full"
        prompt = "Test prompt"
        prompt_token_ids = [1, 2, 3]
        prompt_token_ids_len = 3

        outputs = CompletionOutput(
            index=0, send_idx=0, token_ids=[100, 200], text="Generated text", reasoning_content="Reasoning content"
        )

        metrics = RequestMetrics()
        metrics.arrival_time = time.time()

        request_output = RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs={"test": "logprobs"},
            output_type=1,
            outputs=outputs,
            finished=True,
            metrics=metrics,
            num_cached_tokens=5,
            num_input_image_tokens=2,
            num_input_video_tokens=1,
            error_code=400,
            error_msg="Test error",
            ic_req_data={"internal": "data"},
            prompt_token_ids_len=prompt_token_ids_len,
        )

        self.assertEqual(request_output.request_id, request_id)
        self.assertEqual(request_output.prompt, prompt)
        self.assertEqual(request_output.prompt_token_ids, prompt_token_ids)
        self.assertEqual(request_output.prompt_logprobs, {"test": "logprobs"})
        self.assertEqual(request_output.output_type, 1)
        self.assertEqual(request_output.outputs, outputs)
        self.assertTrue(request_output.finished)
        self.assertEqual(request_output.metrics, metrics)
        self.assertEqual(request_output.num_cached_tokens, 5)
        self.assertEqual(request_output.num_input_image_tokens, 2)
        self.assertEqual(request_output.num_input_video_tokens, 1)
        self.assertEqual(request_output.error_code, 400)
        self.assertEqual(request_output.error_msg, "Test error")
        self.assertEqual(request_output.ic_req_data, {"internal": "data"})
        self.assertEqual(request_output.prompt_token_ids_len, prompt_token_ids_len)


class TestRequestOutputAccumulate(unittest.TestCase):
    """Test case for RequestOutput accumulate method"""

    def setUp(self):
        """Set up test fixtures"""
        self.request_id = "test_request_accumulate"
        self.base_request = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(
                index=0, send_idx=0, token_ids=[100, 200], text="First ", reasoning_content="Reasoning "
            ),
        )

    def test_accumulate_basic_text(self):
        """Test basic text accumulation"""
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], text="second"),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.outputs.text, "First second")
        self.assertEqual(self.base_request.outputs.token_ids, [100, 200, 300])
        self.assertEqual(self.base_request.outputs.index, 0)

    def test_accumulate_reasoning_content(self):
        """Test reasoning content accumulation"""
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], reasoning_content="content"),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.outputs.reasoning_content, "Reasoning content")

    def test_accumulate_completion_tokens(self):
        """Test completion tokens accumulation"""
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], completion_tokens=" tokens"),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.outputs.completion_tokens, " tokens")

    def test_accumulate_tool_calls(self):
        """Test tool calls accumulation"""
        tool_calls = [{"name": "tool2", "arguments": {"param": "value2"}}]
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], tool_calls=tool_calls),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.accumulate_tool_calls, [tool_calls])

    def test_accumulate_multiple_tool_calls(self):
        """Test multiple tool calls accumulation"""
        # Add initial tool call through constructor
        initial_tool_calls = [{"name": "tool1", "arguments": {"param": "value1"}}]
        base_request = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=0, token_ids=[100, 200], tool_calls=initial_tool_calls),
        )

        second_tool_calls = [{"name": "tool2", "arguments": {"param": "value2"}}]
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], tool_calls=second_tool_calls),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        base_request.accumulate(next_output)

        self.assertEqual(base_request.accumulate_tool_calls, [initial_tool_calls, second_tool_calls])

    def test_accumulate_with_metrics(self):
        """Test accumulation including metrics updates"""
        next_output = RequestOutput(
            request_id=self.request_id,
            metrics=RequestMetrics(model_forward_time=1.5, model_execute_time=2.5),
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300], text=" text"),
        )

        # Set up base metrics
        self.base_request.metrics = RequestMetrics()

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.metrics.model_forward_time, 1.5)
        self.assertEqual(self.base_request.metrics.model_execute_time, 2.5)

    def test_accumulate_with_logprobs(self):
        """Test accumulation with logprobs data"""
        # Create LogprobsLists objects - each list corresponds to a different request/position
        initial_top_logprobs = LogprobsLists(
            logprob_token_ids=[[100, 200]],  # First request with 2 token probabilities
            logprobs=[[0.7, 0.6]],  # Corresponding log probabilities
            sampled_token_ranks=[0],  # Rank for the first request
        )

        initial_draft_logprobs = LogprobsLists(
            logprobs=[[0.8, 0.7]],  # Default draft logprobs
            logprob_token_ids=[[150, 250]],  # Default draft token IDs
            sampled_token_ranks=[1],  # Default draft ranks
        )

        # Set up initial logprobs
        self.base_request.outputs.top_logprobs = initial_top_logprobs
        self.base_request.outputs.draft_top_logprobs = initial_draft_logprobs

        # Create next output with new logprobs (representing a new decoding step)
        new_top_logprobs = LogprobsLists(
            logprob_token_ids=[[300, 400, 500]],  # New step with 3 token IDs
            logprobs=[[0.5, 0.4, 0.3]],  # Corresponding log probabilities
            sampled_token_ranks=[1],  # New rank
        )

        new_draft_logprobs = LogprobsLists(
            logprob_token_ids=[[350, 450, 550]],  # New draft token IDs
            logprobs=[[0.6, 0.5, 0.4]],  # New draft log probabilities
            sampled_token_ranks=[2],  # New draft rank
        )

        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(
                index=0,
                send_idx=1,
                token_ids=[600],  # New token
                text=" text",
                top_logprobs=new_top_logprobs,
                draft_top_logprobs=new_draft_logprobs,
            ),
            metrics=RequestMetrics(),
        )

        self.base_request.accumulate(next_output)

        # Verify accumulation adds new rows (requests/positions)
        # After accumulation, we should have 2 rows (initial + new)
        self.assertEqual(len(self.base_request.outputs.top_logprobs.logprob_token_ids), 2)
        self.assertEqual(len(self.base_request.outputs.top_logprobs.logprobs), 2)
        self.assertEqual(len(self.base_request.outputs.top_logprobs.sampled_token_ranks), 2)

        # Check first row remains unchanged
        self.assertEqual(self.base_request.outputs.top_logprobs.logprob_token_ids[0], [100, 200])
        self.assertEqual(self.base_request.outputs.top_logprobs.logprobs[0], [0.7, 0.6])
        self.assertEqual(self.base_request.outputs.top_logprobs.sampled_token_ranks[0], 0)

        # Check second row contains new data
        self.assertEqual(self.base_request.outputs.top_logprobs.logprob_token_ids[1], [300, 400, 500])
        self.assertEqual(self.base_request.outputs.top_logprobs.logprobs[1], [0.5, 0.4, 0.3])
        self.assertEqual(self.base_request.outputs.top_logprobs.sampled_token_ranks[1], 1)

        # Same for draft logprobs
        self.assertEqual(len(self.base_request.outputs.draft_top_logprobs.logprob_token_ids), 2)
        self.assertEqual(len(self.base_request.outputs.draft_top_logprobs.logprobs), 2)
        self.assertEqual(len(self.base_request.outputs.draft_top_logprobs.sampled_token_ranks), 2)

        self.assertEqual(self.base_request.outputs.draft_top_logprobs.logprob_token_ids[0], [150, 250])
        self.assertEqual(self.base_request.outputs.draft_top_logprobs.logprobs[0], [0.8, 0.7])
        self.assertEqual(self.base_request.outputs.draft_top_logprobs.sampled_token_ranks[0], 1)

        self.assertEqual(self.base_request.outputs.draft_top_logprobs.logprob_token_ids[1], [350, 450, 550])
        self.assertEqual(self.base_request.outputs.draft_top_logprobs.logprobs[1], [0.6, 0.5, 0.4])
        self.assertEqual(self.base_request.outputs.draft_top_logprobs.sampled_token_ranks[1], 2)

    def test_accumulate_null_text_handling(self):
        """Test accumulate with null text handling"""
        base_request = RequestOutput(
            request_id=self.request_id, outputs=CompletionOutput(index=0, send_idx=0, token_ids=[100])  # text is None
        )

        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[200], text="new text"),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        base_request.accumulate(next_output)

        self.assertEqual(base_request.outputs.text, "new text")

    def test_accumulate_finished_flag(self):
        """Test that finished flag is OR-ed correctly"""
        next_output = RequestOutput(
            request_id=self.request_id,
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300]),
            finished=True,
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertTrue(self.base_request.finished)

    def test_accumulate_prompt_updates(self):
        """Test that prompt and prompt_token_ids are updated from next_output"""
        next_output = RequestOutput(
            request_id=self.request_id,
            prompt="Updated prompt",
            prompt_token_ids=[999],
            outputs=CompletionOutput(index=0, send_idx=1, token_ids=[300]),
            metrics=RequestMetrics(),  # Add metrics to avoid None attribute access
        )

        self.base_request.accumulate(next_output)

        self.assertEqual(self.base_request.prompt, "Updated prompt")
        self.assertEqual(self.base_request.prompt_token_ids, [999])


class TestRequestOutputFromDict(unittest.TestCase):
    """Test case for RequestOutput from_dict method"""

    def test_from_dict_with_outputs_and_metrics(self):
        """Test from_dict with outputs and metrics dictionaries"""
        test_dict = {
            "request_id": "test_dict_123",
            "prompt": "Dict prompt",
            "prompt_token_ids": [1, 2, 3],
            "finished": True,
            "outputs": {"index": 0, "send_idx": 0, "token_ids": [100, 200], "text": "Dict text"},
            "metrics": {"arrival_time": 1000.0, "model_forward_time": 1.5},
        }

        request_output = RequestOutput.from_dict(test_dict)

        self.assertEqual(request_output.request_id, "test_dict_123")
        self.assertEqual(request_output.prompt, "Dict prompt")
        self.assertEqual(request_output.prompt_token_ids, [1, 2, 3])
        self.assertTrue(request_output.finished)
        self.assertIsInstance(request_output.outputs, CompletionOutput)
        self.assertEqual(request_output.outputs.text, "Dict text")
        self.assertIsInstance(request_output.metrics, RequestMetrics)
        self.assertEqual(request_output.metrics.arrival_time, 1000.0)

    def test_from_dict_without_outputs_and_metrics(self):
        """Test from_dict without outputs and metrics in dictionary"""
        test_dict = {"request_id": "test_dict_456", "finished": False}

        request_output = RequestOutput.from_dict(test_dict)

        self.assertEqual(request_output.request_id, "test_dict_456")
        self.assertFalse(request_output.finished)
        self.assertIsNone(request_output.outputs)
        self.assertIsNone(request_output.metrics)


if __name__ == "__main__":
    unittest.main()
