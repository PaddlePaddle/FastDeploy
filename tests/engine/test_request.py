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
import unittest
from unittest.mock import Mock

import numpy as np

from fastdeploy.engine.request import (
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


if __name__ == "__main__":
    unittest.main()
