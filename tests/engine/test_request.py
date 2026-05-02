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

import json
import pickle
import unittest
from unittest.mock import Mock

import numpy as np

from fastdeploy.cache_manager.v1.metadata import CacheLevel, CacheSwapMetadata
from fastdeploy.engine.request import (
    BatchRequest,
    CompletionOutput,
    ImagePosition,
    PoolingParams,
    Request,
    RequestMetrics,
    RequestOutput,
    RequestStatus,
    RequestType,
    SamplingParams,
    StructuralTagResponseFormat,
)
from fastdeploy.entrypoints.openai.protocol import ResponseFormat, StructuralTag


def _make_swap_meta(src_ids, dst_ids, hash_values=None):
    """Helper: create a CacheSwapMetadata instance."""
    return CacheSwapMetadata(
        src_block_ids=list(src_ids),
        dst_block_ids=list(dst_ids),
        src_type="host",
        dst_type="device",
        hash_values=list(hash_values) if hash_values else [],
    )


class TestRequestInit(unittest.TestCase):
    """Test cases for Request initialization"""

    def test_init_default_values(self):
        """Test initialization with default values"""
        request = Request(request_id="test_123")

        # Test basic attributes
        self.assertEqual(request.request_id, "test_123")
        self.assertIsNone(request.prompt)
        self.assertIsNone(request.prompt_token_ids)
        self.assertIsNone(request.prompt_token_ids_len)
        self.assertIsNone(request.messages)
        self.assertIsNone(request.system)
        self.assertIsNone(request.sampling_params)
        self.assertIsNone(request.pooling_params)
        self.assertIsNone(request.history)
        self.assertIsNone(request.tools)
        self.assertIsNone(request.eos_token_ids)

        # Test default values
        self.assertEqual(request.num_cached_tokens, 0)
        self.assertEqual(request.num_cached_blocks, 0)
        self.assertFalse(request.disable_chat_template)
        self.assertIsNone(request.disaggregate_info)

        # Test multi-modal defaults
        self.assertIsNone(request.multimodal_inputs)
        self.assertIsNone(request.multimodal_data)
        self.assertIsNone(request.multimodal_img_boundaries)

        # Test status and type
        self.assertEqual(request.status, RequestStatus.WAITING)
        self.assertEqual(request.task_type, RequestType.PREFILL)
        self.assertIsNone(request.idx)
        self.assertEqual(request.need_prefill_tokens, None)  # prompt_token_ids_len is None

        # Test internal structures
        self.assertEqual(request.block_tables, [])
        self.assertEqual(request.output_token_ids, [])
        self.assertEqual(request.num_computed_tokens, 0)
        self.assertEqual(request.prefill_start_index, 0)
        self.assertEqual(request.prefill_end_index, 0)
        self.assertEqual(request.async_process_futures, [])
        self.assertIsNone(request.error_message)
        self.assertIsNone(request.error_code)

    def test_init_with_parameters(self):
        """Test initialization with various parameters"""
        sampling_params = SamplingParams()
        pooling_params = PoolingParams()
        metrics = RequestMetrics()

        request = Request(
            request_id="test_full",
            prompt="Hello world",
            prompt_token_ids=[1, 2, 3],
            prompt_token_ids_len=3,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful",
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            history=[["user", "hello"]],
            tools=[{"name": "test_tool"}],
            eos_token_ids=[0],
            disable_chat_template=True,
            disaggregate_info={"key": "value"},
            draft_token_ids=[4, 5],
            guided_json={"schema": "test"},
            guided_regex="test.*",
            guided_choice=["option1", "option2"],
            guided_grammar="grammar",
            structural_tag="tag",
            guided_json_object=True,
            enable_thinking=True,
            reasoning_max_tokens=100,
            trace_carrier={"trace": "carrier"},
            dp_rank=0,
            chat_template="template",
            image_start=1,
            video_start=2,
            audio_start=3,
            image_end=4,
            video_end=5,
            audio_end=6,
            prefill_start_index=10,
            prefill_end_index=20,
            num_computed_tokens=5,
            metrics=metrics,
            user="test_user",
            metadata={"meta": "data"},
            completion_token_ids=[6, 7],
            chat_template_kwargs={"kwarg": "value"},
            prompt_tokens="tokens",
            add_generation_prompt=True,
            response_format={"type": "json_object"},
            mm_hashes=["hash1", "hash2"],
            suffix={"key": "suffix"},
            top_logprobs=5,
            add_special_tokens=True,
        )

        # Test parameter assignment
        self.assertEqual(request.request_id, "test_full")
        self.assertEqual(request.prompt, "Hello world")
        self.assertEqual(request.prompt_token_ids, [1, 2, 3])
        self.assertEqual(request.prompt_token_ids_len, 3)
        self.assertEqual(request.messages, [{"role": "user", "content": "Hello"}])
        self.assertEqual(request.system, "You are helpful")
        self.assertEqual(request.sampling_params, sampling_params)
        self.assertEqual(request.pooling_params, pooling_params)
        self.assertEqual(request.history, [["user", "hello"]])
        self.assertEqual(request.tools, [{"name": "test_tool"}])
        self.assertEqual(request.eos_token_ids, [0])

        # Test boolean parameters
        self.assertTrue(request.disable_chat_template)
        self.assertTrue(request.guided_json_object)
        self.assertTrue(request.enable_thinking)
        self.assertTrue(request.add_generation_prompt)
        self.assertTrue(request.add_special_tokens)

        # Test numerical parameters
        self.assertEqual(request.reasoning_max_tokens, 100)
        self.assertEqual(request.dp_rank, 0)
        self.assertEqual(request.image_start, 1)
        self.assertEqual(request.video_start, 2)

        # Test string parameters
        self.assertEqual(request.trace_carrier, {"trace": "carrier"})
        self.assertEqual(request.chat_template, "template")
        self.assertEqual(request.user, "test_user")

    def test_init_with_multimodal_inputs(self):
        """Test initialization with multimodal inputs"""
        multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=0, length=10)],
            "input_ids": np.array([1, 2, 3]),
        }

        request = Request(
            request_id="test_mm",
            multimodal_inputs=multimodal_inputs,
            multimodal_data={"images": ["img1", "img2"]},
        )

        self.assertEqual(request.multimodal_inputs, multimodal_inputs)
        self.assertEqual(request.multimodal_data, {"images": ["img1", "img2"]})
        self.assertIsNone(request.multimodal_img_boundaries)

    def test_init_default_metrics(self):
        """Test that metrics are created when not provided"""
        request = Request(request_id="test_metrics")
        self.assertIsInstance(request.metrics, RequestMetrics)
        self.assertIsNotNone(request.metrics.arrival_time)

    def test_init_existing_metrics(self):
        """Test initialization with existing metrics"""
        metrics = RequestMetrics()
        metrics.arrival_time = 1000.0

        request = Request(request_id="test_existing_metrics", metrics=metrics)
        self.assertEqual(request.metrics, metrics)
        self.assertEqual(request.metrics.arrival_time, 1000.0)


class TestRequestProperties(unittest.TestCase):
    """Test cases for Request properties"""

    def test_num_total_tokens(self):
        """Test num_total_tokens property"""
        # Test with no tokens
        request = Request(request_id="test1")
        request.prompt_token_ids_len = 0
        self.assertEqual(request.num_total_tokens, 0)

        # Test with prompt tokens only
        request = Request(request_id="test2")
        request.prompt_token_ids_len = 5
        request.output_token_ids = []
        self.assertEqual(request.num_total_tokens, 5)

        # Test with output tokens only
        request = Request(request_id="test3")
        request.prompt_token_ids_len = 0
        request.output_token_ids = [1, 2, 3]
        self.assertEqual(request.num_total_tokens, 3)

        # Test with both prompt and output tokens
        request = Request(request_id="test4")
        request.prompt_token_ids_len = 5
        request.output_token_ids = [1, 2, 3]
        self.assertEqual(request.num_total_tokens, 8)


class TestRequestClassMethods(unittest.TestCase):
    """Test cases for Request class methods"""

    def test_process_guided_json(self):
        """Test _process_guided_json class method"""
        # Test with response_format type json_object
        mock_request = Request(request_id="pickle_test")
        mock_request.response_format = ResponseFormat(type="json_object")
        result = Request._process_guided_json(mock_request)
        self.assertTrue(result)
        self.assertIsNone(getattr(mock_request, "guided_json", None))

        # Test with response_format type json_schema
        mock_request = Mock()
        mock_request.response_format = Mock()
        mock_request.response_format.type = "json_schema"
        mock_request.response_format.json_schema = Mock()
        mock_request.response_format.json_schema.json_schema = {"type": "object"}

        Request._process_guided_json(mock_request)
        self.assertEqual(mock_request.guided_json, {"type": "object"})

        # Test with response_format type structural_tag
        mock_request = Mock()
        mock_request.response_format = StructuralTagResponseFormat(
            type="structural_tag",
            structures=[StructuralTag(begin="<user>", end="</user>")],
            triggers=["<user>", "</user>"],
        )

        Request._process_guided_json(mock_request)
        expected_json = json.dumps(
            {
                "type": "structural_tag",
                "structures": [{"begin": "<user>", "schema": None, "end": "</user>"}],
                "triggers": ["<user>", "</user>"],
            }
        )
        self.assertEqual(mock_request.structural_tag, expected_json)

    def test_from_generic_request(self):
        """Test from_generic_request class method"""
        mock_generic_request = Mock()
        mock_generic_request.request_id = "generic_test"
        mock_generic_request.prompt_token_ids = [1, 2, 3]
        mock_generic_request.messages = [{"role": "user", "content": "Hello"}]
        mock_generic_request.disable_chat_template = True
        mock_generic_request.tools = [Mock()]
        mock_generic_request.tools[0].model_dump.return_value = {"name": "test_tool"}
        mock_generic_request.suffix = {"test": "value"}
        mock_generic_request.metadata = {"key": "value"}

        # Mock sampling params creation
        original_from_generic = SamplingParams.from_generic_request
        SamplingParams.from_generic_request = Mock(return_value=SamplingParams())

        try:
            request = Request.from_generic_request(
                req=mock_generic_request,
                request_id="override_test",
                prompt="Test prompt",
            )

            self.assertEqual(request.request_id, "override_test")
            self.assertEqual(request.prompt, "Test prompt")
            self.assertEqual(request.prompt_token_ids, [1, 2, 3])
            self.assertEqual(request.messages, [{"role": "user", "content": "Hello"}])
            self.assertTrue(request.disable_chat_template)
            self.assertEqual(request.tools, [{"name": "test_tool"}])
            self.assertIsInstance(request.metrics, RequestMetrics)

        finally:
            SamplingParams.from_generic_request = original_from_generic

    def test_from_dict(self):
        """Test from_dict class method"""
        test_dict = {
            "request_id": "dict_test",
            "prompt": "Test prompt",
            "prompt_token_ids": [1, 2, 3],
            "prompt_token_ids_len": 3,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Test system",
            "history": [["user", "hi"]],
            "tools": [{"name": "test_tool"}],
            "eos_token_ids": [0],
            "multimodal_inputs": {"mm_positions": [{"offset": 0, "length": 10}]},
            "multimodal_data": {"images": ["img1"]},
            "disable_chat_template": True,
            "disaggregate_info": {"key": "value"},
            "draft_token_ids": [4, 5],
            "guided_json": {"schema": "test"},
            "guided_regex": "test.*",
            "guided_choice": ["opt1"],
            "guided_grammar": "grammar",
            "structural_tag": "tag",
            "guided_json_object": True,
            "enable_thinking": True,
            "reasoning_max_tokens": 100,
            "trace_carrier": {"trace": "carrier"},
            "chat_template": "template",
            "num_computed_tokens": 5,
            "prefill_start_index": 10,
            "prefill_end_index": 20,
            "image_start": 1,
            "video_start": 2,
            "audio_start": 3,
            "image_end": 4,
            "video_end": 5,
            "audio_end": 6,
            "dp_rank": 0,
            "ic_req_data": {"internal": "data"},
            "metrics": {"arrival_time": 1000.0},
            "max_tokens": 100,
        }

        request = Request.from_dict(test_dict)

        # Test basic fields
        self.assertEqual(request.request_id, "dict_test")
        self.assertEqual(request.prompt, "Test prompt")
        self.assertEqual(request.prompt_token_ids, [1, 2, 3])
        self.assertEqual(request.prompt_token_ids_len, 3)

        # Test multimodal inputs conversion
        self.assertIsInstance(request.multimodal_inputs["mm_positions"][0], ImagePosition)

        # Test sampling params creation
        self.assertIsNotNone(request.sampling_params)

        # Test metrics creation
        self.assertIsInstance(request.metrics, RequestMetrics)
        self.assertEqual(request.metrics.arrival_time, 1000.0)


class TestRequestInstanceMethods(unittest.TestCase):
    """Test cases for Request instance methods"""

    def test_getstate(self):
        """Test __getstate__ method for pickle support"""
        request = Request(request_id="pickle_test")
        request.async_process_futures = [Mock(), Mock()]  # These should be filtered

        state = request.__getstate__()

        # async_process_futures should be empty list after filtering
        self.assertEqual(state["async_process_futures"], [])
        # Other attributes should be preserved
        self.assertEqual(state["request_id"], "pickle_test")

    def test_eq(self):
        """Test __eq__ method"""
        request1 = Request(request_id="same_id")
        request2 = Request(request_id="same_id")
        request3 = Request(request_id="different_id")

        self.assertEqual(request1, request2)
        self.assertNotEqual(request1, request3)
        self.assertNotEqual(request1, "not_a_request")

    def test_to_dict_basic(self):
        """Test to_dict method with basic request"""
        request = Request(request_id="dict_basic")
        request.prompt = "Hello"
        request.prompt_token_ids = [1, 2, 3]
        request.prompt_token_ids_len = 3
        request.sampling_params = SamplingParams()
        request.metrics = RequestMetrics()
        request.metrics.prompt_token_ids_len = 3

        data = request.to_dict()

        self.assertEqual(data["request_id"], "dict_basic")
        self.assertEqual(data["prompt"], "Hello")
        self.assertEqual(data["prompt_token_ids"], [1, 2, 3])
        self.assertEqual(data["prompt_token_ids_len"], 3)

    def test_to_dict_with_multimodal(self):
        """Test to_dict with multimodal inputs"""
        request = Request(request_id="dict_mm")
        request.multimodal_inputs = {
            "position_ids": [1, 2, 3],
            "input_ids": np.array([4, 5, 6]),
            "other_field": "should_be_filtered",
        }
        request.sampling_params = SamplingParams()
        request.metrics = RequestMetrics()

        # Test with V1 scheduler (should only allow position_ids)
        data = request.to_dict()
        self.assertEqual(list(data["multimodal_inputs"].keys()), ["position_ids"])
        self.assertEqual(data["multimodal_inputs"]["position_ids"], [1, 2, 3])

    def test_get_method(self):
        """Test get method for attribute access"""
        request = Request(request_id="get_test")
        request.sampling_params = SamplingParams()
        request.sampling_params.temperature = 0.7

        # Test getting request attribute
        self.assertEqual(request.get("request_id"), "get_test")

        # Test getting sampling_params attribute
        self.assertEqual(request.get("temperature"), 0.7)

        # Test getting non-existent attribute with default
        self.assertIsNone(request.get("non_existent"))
        self.assertEqual(request.get("non_existent", "default"), "default")

    def test_set_method(self):
        """Test set method for attribute modification"""
        request = Request(request_id="set_test")
        request.sampling_params = SamplingParams()

        # Test setting request attribute
        request.set("prompt", "New prompt")
        self.assertEqual(request.prompt, "New prompt")

        # Test setting sampling_params attribute
        request.set("temperature", 1.0)
        self.assertEqual(request.sampling_params.temperature, 1.0)

    def test_repr_debug_disabled(self):
        """Test __repr__ when debug is disabled"""
        request = Request(request_id="repr_test")
        repr_str = request.__repr__()
        self.assertEqual(repr_str, "Request(request_id=repr_test)")

    def test_repr_debug_enabled(self):
        """Test __repr__ when debug is enabled"""
        request = Request(request_id="repr_debug")
        request.prompt = "Hello"
        request.prompt_token_ids = [1, 2, 3]

        # Mock envs.FD_DEBUG to True
        import fastdeploy.engine.request as request_module

        original_value = getattr(request_module.envs, "FD_DEBUG", False)
        request_module.envs.FD_DEBUG = True

        try:
            repr_str = request.__repr__()
            self.assertIn("request_id='repr_debug'", repr_str)
            self.assertIn("prompt='Hello'", repr_str)
            self.assertIn("prompt_token_ids=[1, 2, 3]", repr_str)
        finally:
            request_module.envs.FD_DEBUG = original_value

    def test_getitem_setitem_delitem(self):
        """Test dictionary-like access methods"""
        request = Request(request_id="dict_access")
        request.sampling_params = SamplingParams()
        request.sampling_params.temperature = 0.7

        # Test __getitem__
        self.assertEqual(request["request_id"], "dict_access")
        self.assertEqual(request["temperature"], 0.7)

        # Test __setitem__
        request["prompt"] = "New prompt"
        self.assertEqual(request.prompt, "New prompt")
        request["temperature"] = 1.0
        self.assertEqual(request.sampling_params.temperature, 1.0)

        # Test __delitem__
        request.sampling_params.top_k = 10
        del request["top_k"]
        self.assertNotIn("top_k", request.sampling_params.__dict__)

    def test_contains(self):
        """Test __contains__ method"""
        request = Request(request_id="contains_test")
        request.sampling_params = SamplingParams()
        request.sampling_params.temperature = 0.7

        self.assertTrue("request_id" in request)
        self.assertTrue("temperature" in request)
        self.assertFalse("non_existent" in request)


class TestRequestEdgeCases(unittest.TestCase):
    """Test edge cases and error scenarios"""

    def test_init_with_none_request_id(self):
        """Test initialization with None request_id"""
        request = Request(request_id=None)
        self.assertIsNone(request.request_id)

    def test_getitem_key_error(self):
        """Test __getitem__ with non-existent key raises KeyError"""
        request = Request(request_id="key_error_test")

        with self.assertRaises(KeyError):
            _ = request["non_existent_key"]

    def test_delitem_key_error(self):
        """Test __delitem__ with non-existent key raises KeyError"""
        request = Request(request_id="del_key_error_test")

        with self.assertRaises(KeyError):
            del request["non_existent_key"]

    def test_repr_exception_handling(self):
        """Test __repr__ handles exceptions gracefully"""
        request = Request(request_id="repr_exception")

        # Create an attribute that will cause an exception during repr
        class ProblematicAttribute:
            def __repr__(self):
                raise Exception("Repr failed")

        request.problematic = ProblematicAttribute()

        # Mock envs.FD_DEBUG to True to trigger detailed repr
        import fastdeploy.engine.request as request_module

        original_value = getattr(request_module.envs, "FD_DEBUG", False)
        request_module.envs.FD_DEBUG = True

        try:
            repr_str = request.__repr__()
            self.assertTrue(repr_str.startswith("<Request repr failed:"))
        finally:
            request_module.envs.FD_DEBUG = original_value

    def test_from_dict_error_handling(self):
        """Test from_dict handles errors in multimodal conversion"""
        test_dict = {
            "request_id": "error_test",
            "multimodal_inputs": {"mm_positions": [{"not_valid": "data"}]},  # Missing required fields
        }

        # Should not raise an exception but log error
        request = Request.from_dict(test_dict)
        self.assertEqual(request.request_id, "error_test")


class TestRequestOutputDictAccess(unittest.TestCase):
    """Test cases for RequestOutput dictionary-style access methods"""

    def setUp(self):
        self.metrics = RequestMetrics()
        self.metrics.arrival_time = 1000.0
        self.metrics.model_forward_time = 1.5

        self.outputs = CompletionOutput(
            index=0, send_idx=0, token_ids=[1, 2, 3], text="test output", reasoning_content="test reasoning"
        )

        self.request_output = RequestOutput(
            request_id="test_dict_access",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            outputs=self.outputs,
            metrics=self.metrics,
        )

    def test_get_method(self):
        """Test get() method"""
        # Test getting request_output attribute
        self.assertEqual(self.request_output.get("request_id"), "test_dict_access")

        # Test getting outputs attribute
        self.assertEqual(self.request_output.get("text"), "test output")

        # Test getting metrics attribute
        self.assertEqual(self.request_output.get("arrival_time"), 1000.0)

        # Test getting non-existent attribute with default
        self.assertIsNone(self.request_output.get("non_existent"))
        self.assertEqual(self.request_output.get("non_existent", "default"), "default")

    def test_set_method(self):
        """Test set() method"""
        # Test setting request_output attribute
        self.request_output.set("prompt", "new prompt")
        self.assertEqual(self.request_output.prompt, "new prompt")

        # Test setting outputs attribute
        self.request_output.set("text", "new text")
        self.assertEqual(self.outputs.text, "new text")

        # Test setting metrics attribute
        self.request_output.set("model_forward_time", 2.0)
        self.assertEqual(self.metrics.model_forward_time, 2.0)

    def test_getitem_method(self):
        """Test __getitem__ method"""
        # Test getting request_output attribute
        self.assertEqual(self.request_output["request_id"], "test_dict_access")

        # Test getting outputs attribute
        self.assertEqual(self.request_output["text"], "test output")

        # Test getting metrics attribute
        self.assertEqual(self.request_output["arrival_time"], 1000.0)

        # Test KeyError for non-existent attribute
        with self.assertRaises(KeyError):
            _ = self.request_output["non_existent"]

    def test_setitem_method(self):
        """Test __setitem__ method"""
        # Test setting request_output attribute
        self.request_output["prompt"] = "new prompt"
        self.assertEqual(self.request_output.prompt, "new prompt")

        # Test setting outputs attribute
        self.request_output["text"] = "new text"
        self.assertEqual(self.outputs.text, "new text")

        # Test setting metrics attribute
        self.request_output["model_forward_time"] = 2.0
        self.assertEqual(self.metrics.model_forward_time, 2.0)

    def test_delitem_method(self):
        """Test __delitem__ method"""
        # Test deleting request_output attribute (using existing attribute)
        original_prompt = self.request_output.prompt
        del self.request_output["prompt"]
        self.assertFalse(hasattr(self.request_output, "prompt"))
        # Restore for other tests
        self.request_output.prompt = original_prompt

        # Test deleting outputs attribute (using existing attribute)
        original_text = self.outputs.text
        del self.request_output["text"]
        self.assertFalse(hasattr(self.outputs, "text"))
        # Restore for other tests
        self.outputs.text = original_text

        # Test deleting metrics attribute (using existing attribute)
        original_arrival_time = self.metrics.arrival_time
        del self.request_output["arrival_time"]
        self.assertFalse(hasattr(self.metrics, "arrival_time"))
        # Restore for other tests
        self.metrics.arrival_time = original_arrival_time

        # Test KeyError for non-existent attribute
        try:
            del self.request_output["non_existent"]
            self.fail("Expected KeyError but none was raised")
        except KeyError:
            pass  # Expected behavior

    def test_contains_method(self):
        """Test __contains__ method"""
        # Test request_output attributes
        self.assertTrue("request_id" in self.request_output)
        self.assertTrue("prompt" in self.request_output)

        # Test outputs attributes
        self.assertTrue("text" in self.request_output)
        self.assertTrue("reasoning_content" in self.request_output)

        # Test metrics attributes
        self.assertTrue("arrival_time" in self.request_output)
        self.assertTrue("model_forward_time" in self.request_output)

        # Test non-existent attribute
        self.assertFalse("non_existent" in self.request_output)


class TestRequestCacheFields(unittest.TestCase):
    """Tests for _block_hasher, _prompt_hashes, cache_swap_metadata, cache_evict_metadata."""

    # ------------------------------------------------------------------
    # _block_hasher / _prompt_hashes initialization
    # ------------------------------------------------------------------

    def test_default_block_hasher_and_prompt_hashes(self):
        """Default values: _block_hasher is None, _prompt_hashes is empty list."""
        req = Request(request_id="cache_defaults")
        self.assertIsNone(req._block_hasher)
        self.assertEqual(req._prompt_hashes, [])

    def test_block_hasher_init_via_constructor(self):
        """block_hasher passed to constructor is stored in _block_hasher."""
        hasher = Mock(return_value=[])
        req = Request(request_id="bh_init", block_hasher=hasher)
        self.assertIs(req._block_hasher, hasher)

    def test_set_block_hasher(self):
        """set_block_hasher replaces _block_hasher."""
        req = Request(request_id="set_bh")
        self.assertIsNone(req._block_hasher)
        hasher = Mock(return_value=[])
        req.set_block_hasher(hasher)
        self.assertIs(req._block_hasher, hasher)

    # ------------------------------------------------------------------
    # prompt_hashes property
    # ------------------------------------------------------------------

    def test_prompt_hashes_no_hasher(self):
        """prompt_hashes returns _prompt_hashes as-is when no hasher is set."""
        req = Request(request_id="ph_no_hasher")
        req._prompt_hashes = ["h1", "h2"]
        self.assertEqual(req.prompt_hashes, ["h1", "h2"])

    def test_prompt_hashes_hasher_returns_new_hashes(self):
        """prompt_hashes appends new hashes returned by _block_hasher."""
        req = Request(request_id="ph_new_hashes")
        req._prompt_hashes = ["h1"]
        req._block_hasher = Mock(return_value=["h2", "h3"])
        result = req.prompt_hashes
        # hasher is called with req
        req._block_hasher.assert_called_once_with(req)
        self.assertEqual(result, ["h1", "h2", "h3"])
        # underlying list is mutated
        self.assertEqual(req._prompt_hashes, ["h1", "h2", "h3"])

    def test_prompt_hashes_hasher_returns_empty(self):
        """When hasher returns empty list, _prompt_hashes is unchanged."""
        req = Request(request_id="ph_empty")
        req._prompt_hashes = ["h1"]
        req._block_hasher = Mock(return_value=[])
        result = req.prompt_hashes
        self.assertEqual(result, ["h1"])
        self.assertEqual(req._prompt_hashes, ["h1"])

    def test_prompt_hashes_hasher_returns_none(self):
        """When hasher returns None (falsy), _prompt_hashes is unchanged."""
        req = Request(request_id="ph_none")
        req._prompt_hashes = ["h1"]
        req._block_hasher = Mock(return_value=None)
        result = req.prompt_hashes
        self.assertEqual(result, ["h1"])

    def test_prompt_hashes_accumulates_across_multiple_accesses(self):
        """Each access may add more hashes (simulates incremental computation)."""
        call_count = {"n": 0}

        def incremental_hasher(r):
            call_count["n"] += 1
            return [f"h{call_count['n']}"]

        req = Request(request_id="ph_incremental")
        req._block_hasher = incremental_hasher
        _ = req.prompt_hashes  # first access → adds "h1"
        _ = req.prompt_hashes  # second access → adds "h2"
        self.assertEqual(req._prompt_hashes, ["h1", "h2"])

    # ------------------------------------------------------------------
    # cache_swap_metadata / cache_evict_metadata initialization
    # ------------------------------------------------------------------

    def test_default_cache_metadata_are_empty_lists(self):
        """cache_swap_metadata and cache_evict_metadata default to empty lists."""
        req = Request(request_id="meta_defaults")
        self.assertEqual(req.cache_swap_metadata, [])
        self.assertEqual(req.cache_evict_metadata, [])

    # ------------------------------------------------------------------
    # pop_cache_swap_metadata / pop_cache_evict_metadata
    # ------------------------------------------------------------------

    def test_pop_cache_swap_metadata_returns_and_clears(self):
        """pop_cache_swap_metadata returns current list and resets to []."""
        req = Request(request_id="pop_swap")
        meta = _make_swap_meta([1], [2], ["hash_a"])
        req.cache_swap_metadata = [meta]
        result = req.pop_cache_swap_metadata()
        self.assertEqual(result, [meta])
        self.assertEqual(req.cache_swap_metadata, [])

    def test_pop_cache_evict_metadata_returns_and_clears(self):
        """pop_cache_evict_metadata returns current list and resets to []."""
        req = Request(request_id="pop_evict")
        meta = _make_swap_meta([3], [4], ["hash_b"])
        req.cache_evict_metadata = [meta]
        result = req.pop_cache_evict_metadata()
        self.assertEqual(result, [meta])
        self.assertEqual(req.cache_evict_metadata, [])

    def test_pop_empty_cache_metadata(self):
        """pop on empty list returns [] and leaves field as []."""
        req = Request(request_id="pop_empty")
        self.assertEqual(req.pop_cache_swap_metadata(), [])
        self.assertEqual(req.pop_cache_evict_metadata(), [])

    # ------------------------------------------------------------------
    # __getstate__ skips _block_hasher
    # ------------------------------------------------------------------

    def test_getstate_excludes_block_hasher(self):
        """__getstate__ must not include _block_hasher (cannot be pickled)."""
        req = Request(request_id="getstate_bh", block_hasher=lambda r: [])
        state = req.__getstate__()
        self.assertNotIn("_block_hasher", state)

    def test_getstate_preserves_prompt_hashes(self):
        """__getstate__ preserves _prompt_hashes."""
        req = Request(request_id="getstate_ph")
        req._prompt_hashes = ["h1", "h2"]
        state = req.__getstate__()
        self.assertEqual(state["_prompt_hashes"], ["h1", "h2"])


class TestBatchRequestInit(unittest.TestCase):
    """Tests for BatchRequest initialization."""

    def test_default_init(self):
        """BatchRequest starts with empty requests and no metadata."""
        br = BatchRequest()
        self.assertEqual(br.requests, [])
        self.assertIsNone(br.cache_swap_metadata)
        self.assertIsNone(br.cache_evict_metadata)

    def test_len_empty(self):
        self.assertEqual(len(BatchRequest()), 0)


class TestBatchRequestAddRequest(unittest.TestCase):
    """Tests for BatchRequest.add_request."""

    def _make_request(self, rid):
        return Request(request_id=rid)

    def test_add_request_appends_to_requests(self):
        """add_request stores request in .requests list."""
        br = BatchRequest()
        req = self._make_request("r1")
        br.add_request(req)
        self.assertIn(req, br.requests)
        self.assertEqual(len(br), 1)

    def test_add_request_without_metadata(self):
        """When request has no pending metadata, batch metadata stays None."""
        br = BatchRequest()
        req = self._make_request("r_no_meta")
        br.add_request(req)
        self.assertIsNone(br.cache_swap_metadata)
        self.assertIsNone(br.cache_evict_metadata)

    def test_add_request_with_swap_metadata(self):
        """add_request moves swap metadata from request to batch."""
        br = BatchRequest()
        req = self._make_request("r_swap")
        meta = _make_swap_meta([10, 11], [20, 21], ["hA", "hB"])
        req.cache_swap_metadata = [meta]

        br.add_request(req)

        # Request's swap list should be cleared
        self.assertEqual(req.cache_swap_metadata, [])
        # Batch should aggregate the metadata
        self.assertIsNotNone(br.cache_swap_metadata)
        self.assertEqual(br.cache_swap_metadata.src_block_ids, [10, 11])
        self.assertEqual(br.cache_swap_metadata.dst_block_ids, [20, 21])
        self.assertEqual(br.cache_swap_metadata.hash_values, ["hA", "hB"])

    def test_add_request_with_evict_metadata(self):
        """add_request moves evict metadata from request to batch."""
        br = BatchRequest()
        req = self._make_request("r_evict")
        meta = _make_swap_meta([5], [6], ["hE"])
        req.cache_evict_metadata = [meta]

        br.add_request(req)

        self.assertEqual(req.cache_evict_metadata, [])
        self.assertIsNotNone(br.cache_evict_metadata)
        self.assertEqual(br.cache_evict_metadata.src_block_ids, [5])
        self.assertEqual(br.cache_evict_metadata.dst_block_ids, [6])

    def test_add_multiple_requests_merges_swap_metadata(self):
        """Swap metadata from multiple requests is merged into one."""
        br = BatchRequest()
        for i, (src, dst, h) in enumerate([([1], [2], ["h1"]), ([3], [4], ["h2"])]):
            req = self._make_request(f"r{i}")
            req.cache_swap_metadata = [_make_swap_meta(src, dst, h)]
            br.add_request(req)

        self.assertEqual(br.cache_swap_metadata.src_block_ids, [1, 3])
        self.assertEqual(br.cache_swap_metadata.dst_block_ids, [2, 4])
        self.assertEqual(br.cache_swap_metadata.hash_values, ["h1", "h2"])

    def test_add_multiple_requests_merges_evict_metadata(self):
        """Evict metadata from multiple requests is merged into one."""
        br = BatchRequest()
        for i, (src, dst, h) in enumerate([([7], [8], ["e1"]), ([9], [10], ["e2"])]):
            req = self._make_request(f"re{i}")
            req.cache_evict_metadata = [_make_swap_meta(src, dst, h)]
            br.add_request(req)

        self.assertEqual(br.cache_evict_metadata.src_block_ids, [7, 9])
        self.assertEqual(br.cache_evict_metadata.dst_block_ids, [8, 10])
        self.assertEqual(br.cache_evict_metadata.hash_values, ["e1", "e2"])


class TestBatchRequestAppendSwapEvictMetadata(unittest.TestCase):
    """Unit tests for append_swap_metadata and append_evict_metadata."""

    def test_append_swap_metadata_first_time(self):
        """append_swap_metadata creates CacheSwapMetadata when None."""
        br = BatchRequest()
        meta = _make_swap_meta([1, 2], [3, 4], ["h1", "h2"])
        br.append_swap_metadata([meta])
        self.assertIsNotNone(br.cache_swap_metadata)
        self.assertEqual(br.cache_swap_metadata.src_block_ids, [1, 2])
        self.assertEqual(br.cache_swap_metadata.dst_block_ids, [3, 4])
        self.assertEqual(br.cache_swap_metadata.hash_values, ["h1", "h2"])
        self.assertEqual(br.cache_swap_metadata.src_type, CacheLevel.HOST)
        self.assertEqual(br.cache_swap_metadata.dst_type, CacheLevel.DEVICE)

    def test_append_swap_metadata_merges(self):
        """Subsequent append_swap_metadata extends existing lists."""
        br = BatchRequest()
        br.append_swap_metadata([_make_swap_meta([1], [2], ["hA"])])
        br.append_swap_metadata([_make_swap_meta([3], [4], ["hB"])])
        self.assertEqual(br.cache_swap_metadata.src_block_ids, [1, 3])
        self.assertEqual(br.cache_swap_metadata.dst_block_ids, [2, 4])
        self.assertEqual(br.cache_swap_metadata.hash_values, ["hA", "hB"])

    def test_append_evict_metadata_first_time(self):
        """append_evict_metadata creates CacheSwapMetadata when None."""
        br = BatchRequest()
        meta = _make_swap_meta([5], [6], ["he"])
        br.append_evict_metadata([meta])
        self.assertIsNotNone(br.cache_evict_metadata)
        self.assertEqual(br.cache_evict_metadata.src_block_ids, [5])
        self.assertEqual(br.cache_evict_metadata.dst_block_ids, [6])
        self.assertEqual(br.cache_evict_metadata.dst_type, CacheLevel.HOST)

    def test_append_evict_metadata_merges(self):
        """Subsequent append_evict_metadata extends existing lists."""
        br = BatchRequest()
        br.append_evict_metadata([_make_swap_meta([1], [2], ["e1"])])
        br.append_evict_metadata([_make_swap_meta([3], [4], ["e2"])])
        self.assertEqual(br.cache_evict_metadata.src_block_ids, [1, 3])
        self.assertEqual(br.cache_evict_metadata.dst_block_ids, [2, 4])
        self.assertEqual(br.cache_evict_metadata.hash_values, ["e1", "e2"])

    def test_append_empty_list_is_noop(self):
        """append_swap_metadata / append_evict_metadata with empty list is a no-op."""
        br = BatchRequest()
        br.append_swap_metadata([])
        br.append_evict_metadata([])
        self.assertIsNone(br.cache_swap_metadata)
        self.assertIsNone(br.cache_evict_metadata)


class TestBatchRequestAppendAndExtend(unittest.TestCase):
    """Tests for BatchRequest.append and BatchRequest.extend."""

    def _br_with_swap(self, src, dst, hashes=None):
        br = BatchRequest()
        br.append_swap_metadata([_make_swap_meta(src, dst, hashes or [])])
        return br

    def _br_with_evict(self, src, dst, hashes=None):
        br = BatchRequest()
        br.append_evict_metadata([_make_swap_meta(src, dst, hashes or [])])
        return br

    def test_append_merges_requests(self):
        br1 = BatchRequest()
        br1.add_request(Request(request_id="a"))
        br2 = BatchRequest()
        br2.add_request(Request(request_id="b"))
        br1.append(br2)
        self.assertEqual(len(br1), 2)

    def test_append_merges_swap_metadata(self):
        br1 = self._br_with_swap([1], [2], ["h1"])
        br2 = self._br_with_swap([3], [4], ["h2"])
        br1.append(br2)
        self.assertEqual(br1.cache_swap_metadata.src_block_ids, [1, 3])
        self.assertEqual(br1.cache_swap_metadata.hash_values, ["h1", "h2"])

    def test_append_merges_evict_metadata(self):
        br1 = self._br_with_evict([5], [6], ["e1"])
        br2 = self._br_with_evict([7], [8], ["e2"])
        br1.append(br2)
        self.assertEqual(br1.cache_evict_metadata.src_block_ids, [5, 7])

    def test_append_batch_without_metadata_does_not_create_metadata(self):
        br1 = BatchRequest()
        br1.add_request(Request(request_id="x"))
        br2 = BatchRequest()
        br2.add_request(Request(request_id="y"))
        br1.append(br2)
        self.assertIsNone(br1.cache_swap_metadata)
        self.assertIsNone(br1.cache_evict_metadata)

    def test_extend_multiple_batches(self):
        br_main = BatchRequest()
        sub1 = self._br_with_swap([1], [2], ["h1"])
        sub1.add_request(Request(request_id="s1"))
        sub2 = self._br_with_swap([3], [4], ["h2"])
        sub2.add_request(Request(request_id="s2"))
        br_main.extend([sub1, sub2])
        self.assertEqual(len(br_main), 2)
        self.assertEqual(br_main.cache_swap_metadata.src_block_ids, [1, 3])


class TestBatchRequestIterAndAccess(unittest.TestCase):
    """Tests for __iter__, __getitem__, __len__, __repr__."""

    def _populated_br(self):
        br = BatchRequest()
        for i in range(3):
            br.add_request(Request(request_id=f"r{i}"))
        return br

    def test_iter(self):
        br = self._populated_br()
        ids = [req.request_id for req in br]
        self.assertEqual(ids, ["r0", "r1", "r2"])

    def test_getitem(self):
        br = self._populated_br()
        self.assertEqual(br[0].request_id, "r0")
        self.assertEqual(br[2].request_id, "r2")

    def test_len(self):
        br = self._populated_br()
        self.assertEqual(len(br), 3)

    def test_repr_contains_swap_and_evict(self):
        br = BatchRequest()
        br.append_swap_metadata([_make_swap_meta([1], [2], ["hR"])])
        r = repr(br)
        self.assertIn("BatchRequest", r)
        self.assertIn("swap_metadata", r)
        self.assertIn("evict_metadata", r)


class TestBatchRequestPickle(unittest.TestCase):
    """Ensure BatchRequest can be serialized / deserialized via pickle."""

    def test_pickle_without_block_hasher(self):
        """BatchRequest with plain Requests (no block_hasher) round-trips via pickle."""
        br = BatchRequest()
        req = Request(request_id="pk1", prompt="hello")
        req._prompt_hashes = ["h1"]
        br.add_request(req)
        br.append_swap_metadata([_make_swap_meta([10], [20], ["hP"])])

        data = pickle.dumps(br)
        br2 = pickle.loads(data)

        self.assertEqual(len(br2), 1)
        self.assertEqual(br2[0].request_id, "pk1")
        self.assertEqual(br2.cache_swap_metadata.src_block_ids, [10])

    def test_getstate_skips_block_hasher_in_requests(self):
        """__getstate__ of BatchRequest serializes requests without _block_hasher."""
        br = BatchRequest()
        req = Request(request_id="gs1", block_hasher=lambda r: ["h_new"])
        br.add_request(req)
        state = br.__getstate__()
        # Each request dict must not contain _block_hasher
        for req_state in state["requests"]:
            self.assertNotIn("_block_hasher", req_state)


from fastdeploy.cache_manager.v1.cache_utils import (
    get_block_hash_extra_keys as _get_block_hash_extra_keys,
)
from fastdeploy.cache_manager.v1.cache_utils import (
    get_request_block_hasher as _get_request_block_hasher,
)
from fastdeploy.cache_manager.v1.cache_utils import (
    hash_block_tokens as _hash_block_tokens,
)


class TestPromptHashesWithRealHasher(unittest.TestCase):
    """
    Test Request.prompt_hashes together with the real get_request_block_hasher
    and get_block_hash_extra_keys implementations.

    These tests do NOT use mock hashers, so they exercise the full hash
    computation path (hash_block_tokens → SHA-256 chained hash).
    """

    BLOCK_SIZE = 4  # small block size makes tests easy to reason about

    get_request_block_hasher = staticmethod(_get_request_block_hasher)
    get_block_hash_extra_keys = staticmethod(_get_block_hash_extra_keys)
    hash_block_tokens = staticmethod(_hash_block_tokens)

    def _hasher(self):
        return _get_request_block_hasher(self.BLOCK_SIZE)

    # ------------------------------------------------------------------
    # Basic hash computation
    # ------------------------------------------------------------------

    def test_no_complete_block_returns_empty(self):
        """Fewer tokens than one block → prompt_hashes returns []."""
        req = Request(
            request_id="real_partial", prompt_token_ids=[1, 2, 3], block_hasher=self._hasher()  # < BLOCK_SIZE=4
        )
        self.assertEqual(req.prompt_hashes, [])

    def test_exactly_one_block(self):
        """Exactly block_size tokens → one hash produced."""
        tokens = [10, 20, 30, 40]  # 4 tokens == BLOCK_SIZE
        req = Request(request_id="real_one_block", prompt_token_ids=tokens, block_hasher=self._hasher())
        hashes = req.prompt_hashes
        self.assertEqual(len(hashes), 1)

        # Verify hash value matches hash_block_tokens directly
        expected = self.hash_block_tokens(tokens, None, None)
        self.assertEqual(hashes[0], expected)

    def test_two_complete_blocks(self):
        """Two full blocks → two chained hashes."""
        tokens = list(range(8))  # 8 tokens = 2 blocks of 4
        req = Request(request_id="real_two_blocks", prompt_token_ids=tokens, block_hasher=self._hasher())
        hashes = req.prompt_hashes
        self.assertEqual(len(hashes), 2)

        h0 = self.hash_block_tokens(tokens[:4], None, None)
        h1 = self.hash_block_tokens(tokens[4:8], h0, None)
        self.assertEqual(hashes[0], h0)
        self.assertEqual(hashes[1], h1)

    def test_partial_tail_not_hashed(self):
        """9 tokens with block_size=4 → only 2 complete blocks hashed."""
        tokens = list(range(9))
        req = Request(request_id="real_tail", prompt_token_ids=tokens, block_hasher=self._hasher())
        self.assertEqual(len(req.prompt_hashes), 2)

    def test_hash_is_deterministic(self):
        """Same tokens always produce the same hash."""
        tokens = [1, 2, 3, 4]
        req1 = Request(request_id="det1", prompt_token_ids=tokens, block_hasher=self._hasher())
        req2 = Request(request_id="det2", prompt_token_ids=tokens, block_hasher=self._hasher())
        self.assertEqual(req1.prompt_hashes, req2.prompt_hashes)

    def test_different_tokens_different_hash(self):
        """Different token sequences yield different hashes."""
        req1 = Request(request_id="diff1", prompt_token_ids=[1, 2, 3, 4], block_hasher=self._hasher())
        req2 = Request(request_id="diff2", prompt_token_ids=[5, 6, 7, 8], block_hasher=self._hasher())
        self.assertNotEqual(req1.prompt_hashes, req2.prompt_hashes)

    # ------------------------------------------------------------------
    # Incremental (multi-access) behaviour
    # ------------------------------------------------------------------

    def test_incremental_hashing_does_not_recompute(self):
        """
        If existing hashes already cover N blocks, prompt_hashes only computes
        the next block – not all blocks from scratch.
        """
        tokens = list(range(12))  # 3 blocks of 4
        req = Request(request_id="incremental", prompt_token_ids=tokens, block_hasher=self._hasher())

        # First access: all three blocks computed
        h_all = req.prompt_hashes[:]  # copy
        self.assertEqual(len(h_all), 3)

        # If we artificially reset and call again, hasher sees existing 3 hashes
        # and returns [] because start_token_idx = 3*4 = 12 = num_tokens → no new block
        result2 = req.prompt_hashes
        self.assertEqual(len(result2), 3)  # no duplicates

    def test_new_output_tokens_trigger_additional_hashes(self):
        """
        After output tokens are appended, a second call to prompt_hashes
        produces more hashes (because the combined token sequence now has
        more complete blocks).
        """
        # Start with exactly 1 block of prompt tokens
        tokens = list(range(4))
        req = Request(request_id="out_tokens", prompt_token_ids=tokens, block_hasher=self._hasher())
        req.output_token_ids = []

        first = req.prompt_hashes[:]
        self.assertEqual(len(first), 1)

        # Append 4 output tokens → now 2 complete blocks total
        req.output_token_ids = list(range(4, 8))
        second = req.prompt_hashes[:]
        self.assertEqual(len(second), 2)
        self.assertEqual(second[0], first[0])  # first hash unchanged

    # ------------------------------------------------------------------
    # get_block_hash_extra_keys via prompt_hashes (multimodal path)
    # ------------------------------------------------------------------

    def test_prompt_hashes_no_multimodal_inputs(self):
        """
        With no multimodal_inputs, get_block_hash_extra_keys returns empty
        extra_keys → hash equals plain hash_block_tokens with extra_keys=None.
        """
        tokens = [1, 2, 3, 4]
        req = Request(request_id="mm_none", prompt_token_ids=tokens, block_hasher=self._hasher())
        req.multimodal_inputs = None

        hashes = req.prompt_hashes
        expected = self.hash_block_tokens(tokens, None, None)
        self.assertEqual(hashes[0], expected)

    def test_prompt_hashes_with_multimodal_fully_within_block(self):
        """
        A multimodal item fully within the block contributes its hash as
        extra_keys, changing the computed block hash.
        """
        tokens = [1, 2, 3, 4]
        mm_hash = "img_hash_abc"
        # Image fully within block [0, 4)
        req = Request(request_id="mm_within", prompt_token_ids=tokens, block_hasher=self._hasher())
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=1, length=2)],
            "mm_hashes": [mm_hash],
        }

        hashes = req.prompt_hashes
        # Expected: extra_keys = (mm_hash,)
        expected = self.hash_block_tokens(tokens, None, (mm_hash,))
        self.assertEqual(hashes[0], expected)

    def test_prompt_hashes_multimodal_outside_block_not_included(self):
        """
        A multimodal item that starts after the block end must NOT be included
        in extra_keys for that block.
        """
        tokens = list(range(8))  # 2 blocks: [0,4) and [4,8)
        mm_hash = "img_hash_xyz"
        # Image sits in the second block [4, 8)
        req = Request(request_id="mm_outside", prompt_token_ids=tokens, block_hasher=self._hasher())
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=4, length=2)],
            "mm_hashes": [mm_hash],
        }

        hashes = req.prompt_hashes

        # First block has no multimodal item → extra_keys = None
        h0_expected = self.hash_block_tokens(list(range(4)), None, None)
        self.assertEqual(hashes[0], h0_expected)

        # Second block contains the image
        h1_expected = self.hash_block_tokens(list(range(4, 8)), h0_expected, (mm_hash,))
        self.assertEqual(hashes[1], h1_expected)

    def test_prompt_hashes_multimodal_spanning_two_blocks(self):
        """
        A multimodal item spanning two blocks contributes its hash to each block.
        """
        tokens = list(range(8))
        mm_hash = "span_hash"
        # Image [2, 6) spans both block [0,4) and [4,8)
        req = Request(request_id="mm_span", prompt_token_ids=tokens, block_hasher=self._hasher())
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=2, length=4)],
            "mm_hashes": [mm_hash],
        }

        hashes = req.prompt_hashes
        self.assertEqual(len(hashes), 2)
        # Both blocks include the mm hash as extra_keys
        h0_expected = self.hash_block_tokens(list(range(4)), None, (mm_hash,))
        self.assertEqual(hashes[0], h0_expected)
        h1_expected = self.hash_block_tokens(list(range(4, 8)), h0_expected, (mm_hash,))
        self.assertEqual(hashes[1], h1_expected)

    # ------------------------------------------------------------------
    # get_block_hash_extra_keys direct unit tests
    # ------------------------------------------------------------------

    def test_extra_keys_no_multimodal(self):
        """No multimodal_inputs → empty extra keys."""
        req = Request(request_id="ek_none")
        req.multimodal_inputs = None
        next_idx, keys = self.get_block_hash_extra_keys(req, 0, 4, 0)
        self.assertEqual(keys, [])
        self.assertEqual(next_idx, 0)

    def test_extra_keys_item_fully_inside_block(self):
        """Multimodal item fully inside [start, end) → its hash is collected."""
        req = Request(request_id="ek_inside")
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=1, length=2)],  # [1, 3)
            "mm_hashes": ["hash_inside"],
        }
        next_idx, keys = self.get_block_hash_extra_keys(req, 0, 4, 0)
        self.assertIn("hash_inside", keys)

    def test_extra_keys_item_starts_after_block(self):
        """Multimodal item starts after block end → not included."""
        req = Request(request_id="ek_after")
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=5, length=2)],  # after block [0,4)
            "mm_hashes": ["hash_after"],
        }
        _, keys = self.get_block_hash_extra_keys(req, 0, 4, 0)
        self.assertEqual(keys, [])

    def test_extra_keys_item_ends_before_block(self):
        """Multimodal item ends before block start → fast-exit, not included."""
        req = Request(request_id="ek_before")
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=0, length=1)],  # [0,1) ends before block [2,6)
            "mm_hashes": ["hash_before"],
        }
        _, keys = self.get_block_hash_extra_keys(req, 2, 6, 0)
        self.assertEqual(keys, [])

    def test_extra_keys_item_spans_beyond_block(self):
        """Multimodal item spanning beyond block end → included, and mm_idx points to it."""
        req = Request(request_id="ek_span")
        req.multimodal_inputs = {
            "mm_positions": [ImagePosition(offset=2, length=4)],  # [2, 6) spans [0,4) end
            "mm_hashes": ["hash_span"],
        }
        next_idx, keys = self.get_block_hash_extra_keys(req, 0, 4, 0)
        self.assertIn("hash_span", keys)
        self.assertEqual(next_idx, 0)  # mm_idx points back at the spanning item

    def test_extra_keys_multiple_items_only_overlapping_included(self):
        """Only multimodal items that overlap [start, end) are included."""
        req = Request(request_id="ek_multi")
        req.multimodal_inputs = {
            "mm_positions": [
                ImagePosition(offset=0, length=2),  # [0,2) → in block [0,4): YES
                ImagePosition(offset=2, length=2),  # [2,4) → in block [0,4): YES
                ImagePosition(offset=5, length=2),  # [5,7) → after block [0,4): NO
            ],
            "mm_hashes": ["hA", "hB", "hC"],
        }
        _, keys = self.get_block_hash_extra_keys(req, 0, 4, 0)
        self.assertIn("hA", keys)
        self.assertIn("hB", keys)
        self.assertNotIn("hC", keys)


if __name__ == "__main__":
    unittest.main()
