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

"""Unit tests for MMProcessor base class, data classes, and _CacheClient."""

import pickle
import unittest
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import numpy as np

from fastdeploy.input.multimodal.mm_processor import (
    _DEFAULT_MM_LIMITS,
    MMContext,
    MMItem,
    MMProcessor,
    TokenizationPath,
    _CacheClient,
)
from fastdeploy.input.utils import IDS_TYPE_FLAG

# ------------------------------------------------------------------
# Concrete subclass for testing abstract base class methods
# ------------------------------------------------------------------


class _ConcreteProcessor(MMProcessor):
    """Minimal concrete implementation for testing base class logic."""

    image_placeholder = "<image>"
    video_placeholder = "<video>"
    image_token_str = "<image>"
    video_token_str = "<video>"

    def preprocess_image(self, img, outputs, uuid, token_len=None):
        outputs["images"].append(np.ones((4, 3)))
        outputs["grid_thw"].append(np.array([1, 2, 2]))
        outputs["image_type_ids"].extend([1] * 1)
        outputs["mm_positions"].append(("img_pos", uuid))

    def preprocess_cached_image(self, img_cache, outputs, uuid, token_len=None):
        pixel_values, meta = img_cache
        outputs["images"].append(pixel_values)
        outputs["grid_thw"].append(np.array([meta.get("thw", (1, 2, 2))]))
        outputs["image_type_ids"].extend([1] * 1)
        outputs["mm_positions"].append(("cached_img_pos", uuid))

    def preprocess_video(self, frames, outputs, uuid, token_len=None, meta=None):
        outputs["images"].append(np.ones((8, 3)))
        outputs["grid_thw"].append(np.array([2, 2, 2]))
        outputs["image_type_ids"].extend([2] * 1)
        outputs["mm_positions"].append(("vid_pos", uuid))

    def preprocess_cached_video(self, frames_cache, outputs, uuid, token_len=None):
        pixel_values, meta = frames_cache
        outputs["images"].append(pixel_values)
        outputs["grid_thw"].append(np.array([meta.get("thw", (2, 2, 2))]))
        outputs["image_type_ids"].extend([2] * 1)
        outputs["mm_positions"].append(("cached_vid_pos", uuid))

    def load_video(self, url, item: dict) -> Tuple[Any, dict]:
        return np.zeros((4, 224, 224, 3)), {"fps": 2.0}

    def add_text_positions(self, outputs, num_tokens):
        cur = outputs["cur_position"]
        outputs["position_ids"].extend(list(range(cur, cur + num_tokens)))
        outputs["cur_position"] = cur + num_tokens

    def pack_position_ids(self, outputs):
        outputs["position_ids"] = np.array(outputs["position_ids"], dtype=np.int64)

    @staticmethod
    def mm_num_tokens(grid_thw) -> int:
        if isinstance(grid_thw, list) and len(grid_thw) == 3:
            t, h, w = grid_thw
            return t * h * w // 4
        return 0

    def append_completion_tokens(self, multimodal_inputs, completion_token_ids):
        multimodal_inputs["input_ids"].extend(completion_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * len(completion_token_ids))
        self.add_text_positions(multimodal_inputs, len(completion_token_ids))


def _make_processor(**overrides):
    """Create a _ConcreteProcessor with mocked tokenizer, bypassing __init__."""
    with patch.object(_ConcreteProcessor, "__init__", return_value=None):
        proc = _ConcreteProcessor.__new__(_ConcreteProcessor)

    proc.tokenizer = MagicMock()
    proc.tokenizer.tokenize.return_value = ["hello", "world"]
    proc.tokenizer.convert_tokens_to_ids.return_value = [10, 20]
    proc._cache = None
    proc.enable_processor_cache = False
    proc.image_placeholder = "<image>"
    proc.video_placeholder = "<video>"
    proc.image_token_str = "<image>"
    proc.video_token_str = "<video>"
    proc.limit_mm_per_prompt = dict(_DEFAULT_MM_LIMITS)
    proc.model_name_or_path = "test-model"
    proc.config = None
    proc.fps = 2.0
    proc.min_frames = 4
    proc.max_frames = 768
    proc.target_frames = -1
    proc.role_prefixes = {"system": "", "user": "User: ", "assistant": "Assistant: "}

    for k, v in overrides.items():
        setattr(proc, k, v)
    return proc


# ==================================================================
# Test classes
# ==================================================================


class TestDataClasses(unittest.TestCase):
    """Tests for TokenizationPath, MMItem, MMContext."""

    def test_tokenization_path_values(self):
        self.assertEqual(TokenizationPath.PRETOKENIZED.value, "pretokenized")
        self.assertEqual(TokenizationPath.FROM_TEXT.value, "from_text")

    def test_mm_item_defaults(self):
        item = MMItem(type="image")
        self.assertEqual(item.type, "image")
        self.assertIsNone(item.data)
        self.assertIsNone(item.uuid)

    def test_mm_item_with_values(self):
        item = MMItem(type="video", data="some_data", uuid="abc123")
        self.assertEqual(item.type, "video")
        self.assertEqual(item.data, "some_data")
        self.assertEqual(item.uuid, "abc123")

    def test_mm_context_defaults(self):
        ctx = MMContext()
        self.assertEqual(ctx.images, [])
        self.assertEqual(ctx.videos, [])
        self.assertEqual(ctx.mm_order, [])
        self.assertEqual(ctx.path, TokenizationPath.FROM_TEXT)
        self.assertIsNone(ctx.prompt_token_ids)

    def test_mm_context_with_values(self):
        img = MMItem(type="image", data="img_data")
        vid = MMItem(type="video", data="vid_data")
        ctx = MMContext(
            images=[img],
            videos=[vid],
            mm_order=["image", "video"],
            path=TokenizationPath.PRETOKENIZED,
            prompt_token_ids=[1, 2, 3],
        )
        self.assertEqual(len(ctx.images), 1)
        self.assertEqual(len(ctx.videos), 1)
        self.assertEqual(ctx.mm_order, ["image", "video"])
        self.assertEqual(ctx.path, TokenizationPath.PRETOKENIZED)
        self.assertEqual(ctx.prompt_token_ids, [1, 2, 3])


class TestCacheClient(unittest.TestCase):
    """Tests for _CacheClient ZMQ interactions."""

    @patch("fastdeploy.input.multimodal.mm_processor.zmq")
    def test_lazy_socket_creation(self, mock_zmq):
        client = _CacheClient()
        self.assertIsNone(client._socket)
        # Access socket property triggers creation
        _ = client.socket
        mock_zmq.Context.assert_called_once()
        mock_zmq.Context().socket.assert_called_once_with(mock_zmq.DEALER)

    @patch("fastdeploy.input.multimodal.mm_processor.zmq")
    def test_get_sends_and_receives(self, mock_zmq):
        mock_socket = MagicMock()
        mock_zmq.Context().socket.return_value = mock_socket
        expected_items = [("pixel", {"thw": (1, 2, 2)})]
        mock_socket.recv_multipart.return_value = [b"", pickle.dumps(expected_items)]

        client = _CacheClient()
        result = client.get(["hash1", "hash2"])

        mock_socket.send_multipart.assert_called_once()
        sent_data = mock_socket.send_multipart.call_args[0][0]
        self.assertEqual(sent_data[0], b"")
        self.assertEqual(pickle.loads(sent_data[1]), ["hash1", "hash2"])
        self.assertEqual(result, expected_items)

    @patch("fastdeploy.input.multimodal.mm_processor.zmq")
    def test_get_empty_hashes(self, mock_zmq):
        mock_socket = MagicMock()
        mock_zmq.Context().socket.return_value = mock_socket
        mock_socket.recv_multipart.return_value = [b"", pickle.dumps([])]

        client = _CacheClient()
        result = client.get([])
        self.assertEqual(result, [])

    @patch("fastdeploy.input.multimodal.mm_processor.zmq")
    def test_put_sends_pickled_data(self, mock_zmq):
        mock_socket = MagicMock()
        mock_zmq.Context().socket.return_value = mock_socket

        client = _CacheClient()
        client.put(["h1"], [("pixels", {})])

        mock_socket.send_multipart.assert_called_once()
        sent_data = mock_socket.send_multipart.call_args[0][0]
        self.assertEqual(sent_data[0], b"")
        hashes, items = pickle.loads(sent_data[1])
        self.assertEqual(hashes, ["h1"])
        self.assertEqual(items, [("pixels", {})])


class TestResolveMmData(unittest.TestCase):
    """Tests for MMProcessor._resolve_mm_data."""

    def setUp(self):
        self.proc = _make_processor()

    def test_no_input_raises(self):
        with self.assertRaises(ValueError):
            self.proc._resolve_mm_data({})

    def test_from_text_path(self):
        request = {
            "prompt": "Hello <image>",
            "multimodal_data": {"image": [{"data": "img", "uuid": "u1"}], "mm_order": ["image"]},
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.path, TokenizationPath.FROM_TEXT)
        self.assertEqual(len(ctx.images), 1)
        self.assertEqual(ctx.images[0].data, "img")
        self.assertEqual(ctx.images[0].uuid, "u1")

    def test_pretokenized_path(self):
        request = {
            "prompt_token_ids": [1, 2, 3],
            "multimodal_data": {"image": [{"data": "img", "uuid": "u1"}], "mm_order": ["image"]},
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.path, TokenizationPath.PRETOKENIZED)
        self.assertEqual(ctx.prompt_token_ids, [1, 2, 3])

    def test_dict_images_parsed(self):
        request = {
            "prompt": "test",
            "multimodal_data": {"image": [{"data": "img_data", "uuid": "img_uuid"}], "mm_order": ["image"]},
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.images[0].type, "image")
        self.assertEqual(ctx.images[0].data, "img_data")
        self.assertEqual(ctx.images[0].uuid, "img_uuid")

    def test_raw_images_parsed(self):
        """Non-dict images (e.g., PIL Image) get uuid=None."""
        raw_img = MagicMock()  # simulates PIL Image
        request = {"prompt": "test", "multimodal_data": {"image": [raw_img], "mm_order": ["image"]}}
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.images[0].data, raw_img)
        self.assertIsNone(ctx.images[0].uuid)

    def test_mm_order_from_request(self):
        request = {
            "prompt": "test",
            "multimodal_data": {
                "image": [{"data": "i1", "uuid": "u1"}],
                "video": [{"data": "v1", "uuid": "u2"}],
                "mm_order": ["video", "image"],
            },
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.mm_order, ["video", "image"])

    def test_mm_order_missing_raises(self):
        """Without mm_order, raises ValueError when images/videos exist."""
        request = {
            "prompt": "test",
            "multimodal_data": {
                "image": [{"data": "i1", "uuid": "u1"}],
                "video": [{"data": "v1", "uuid": "u2"}],
            },
        }
        with self.assertRaises(ValueError):
            self.proc._resolve_mm_data(request)

    def test_mm_order_missing_no_mm_items_ok(self):
        """Without mm_order but no images/videos, defaults to [] (pure text)."""
        request = {
            "prompt": "test",
            "multimodal_data": {},
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.mm_order, [])

    def test_videos_parsed(self):
        request = {
            "prompt": "test",
            "multimodal_data": {"video": [{"data": "vid_data", "uuid": "vid_uuid"}], "mm_order": ["video"]},
        }
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(len(ctx.videos), 1)
        self.assertEqual(ctx.videos[0].type, "video")
        self.assertEqual(ctx.videos[0].data, "vid_data")
        self.assertEqual(ctx.videos[0].uuid, "vid_uuid")

    def test_raw_video_parsed(self):
        raw_vid = "http://example.com/video.mp4"
        request = {"prompt": "test", "multimodal_data": {"video": [raw_vid], "mm_order": ["video"]}}
        ctx = self.proc._resolve_mm_data(request)
        self.assertEqual(ctx.videos[0].data, raw_vid)
        self.assertIsNone(ctx.videos[0].uuid)


class TestFetchFromCache(unittest.TestCase):
    """Tests for MMProcessor._fetch_from_cache."""

    def setUp(self):
        self.proc = _make_processor()

    def test_all_data_present_skips_cache(self):
        """When all items have data, cache is not called."""
        ctx = MMContext(
            images=[MMItem(type="image", data="actual_data", uuid="u1")],
            mm_order=["image"],
        )
        self.proc._fetch_from_cache(ctx)
        # No error, no cache interaction

    def test_no_cache_all_data_present_ok(self):
        """No cache configured but data is complete - should work fine."""
        self.proc._cache = None
        ctx = MMContext(
            images=[MMItem(type="image", data="data", uuid="u1")],
            mm_order=["image"],
        )
        self.proc._fetch_from_cache(ctx)  # No error

    def test_no_cache_with_missing_data_raises(self):
        """Missing data without cache raises ValueError."""
        self.proc._cache = None
        ctx = MMContext(
            images=[MMItem(type="image", data=None, uuid="u1")],
            mm_order=["image"],
        )
        with self.assertRaises(ValueError):
            self.proc._fetch_from_cache(ctx)

    def test_cache_hit_fills_data(self):
        mock_cache = MagicMock()
        cached_data = (np.ones((4, 3)), {"thw": (1, 2, 2)})
        mock_cache.get.return_value = [cached_data]
        self.proc._cache = mock_cache

        item = MMItem(type="image", data=None, uuid="hash1")
        ctx = MMContext(images=[item], mm_order=["image"])
        self.proc._fetch_from_cache(ctx)

        mock_cache.get.assert_called_once_with(["hash1"])
        self.assertEqual(item.data, cached_data)

    def test_cache_miss_raises(self):
        mock_cache = MagicMock()
        mock_cache.get.return_value = [None]  # cache miss
        self.proc._cache = mock_cache

        item = MMItem(type="image", data=None, uuid="hash1")
        ctx = MMContext(images=[item], mm_order=["image"])
        with self.assertRaises(ValueError):
            self.proc._fetch_from_cache(ctx)


class TestBuildOutputsFromText(unittest.TestCase):
    """Tests for MMProcessor._build_outputs_from_text."""

    def setUp(self):
        self.proc = _make_processor()

    def test_text_only(self):
        ctx = MMContext(images=[], videos=[], mm_order=[])
        outputs = self.proc._build_outputs_from_text("Hello world", ctx)
        self.assertEqual(outputs["input_ids"], [10, 20])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 2)

    def test_image_placeholder(self):
        img = MMItem(type="image", data="raw_img", uuid="u1")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        outputs = self.proc._build_outputs_from_text("Before<image>After", ctx)
        # preprocess_image was called (appended to images)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(outputs["mm_positions"][0], ("img_pos", "u1"))

    def test_video_placeholder(self):
        vid = MMItem(type="video", data="http://video.mp4", uuid="v1")
        ctx = MMContext(images=[], videos=[vid], mm_order=["video"])
        outputs = self.proc._build_outputs_from_text("Before<video>After", ctx)
        self.assertEqual(len(outputs["images"]), 1)
        self.assertEqual(outputs["mm_positions"][0], ("vid_pos", "v1"))

    def test_video_dict_data(self):
        vid = MMItem(type="video", data={"video": "http://v.mp4", "fps": 1.0}, uuid="v1")
        ctx = MMContext(images=[], videos=[vid], mm_order=["video"])
        outputs = self.proc._build_outputs_from_text("<video>", ctx)
        self.assertEqual(len(outputs["images"]), 1)

    def test_cached_image(self):
        cached = (np.ones((4, 3)), {"thw": (1, 2, 2)})
        img = MMItem(type="image", data=cached, uuid="u1")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        outputs = self.proc._build_outputs_from_text("<image>", ctx)
        self.assertEqual(outputs["mm_positions"][0], ("cached_img_pos", "u1"))

    def test_cached_video(self):
        cached = (np.ones((8, 3)), {"thw": (2, 2, 2)})
        vid = MMItem(type="video", data=cached, uuid="v1")
        ctx = MMContext(images=[], videos=[vid], mm_order=["video"])
        outputs = self.proc._build_outputs_from_text("<video>", ctx)
        self.assertEqual(outputs["mm_positions"][0], ("cached_vid_pos", "v1"))

    def test_interleaved_image_video(self):
        img = MMItem(type="image", data="raw_img", uuid="i1")
        vid = MMItem(type="video", data="http://v.mp4", uuid="v1")
        ctx = MMContext(images=[img], videos=[vid], mm_order=["image", "video"])
        outputs = self.proc._build_outputs_from_text("<image>text<video>", ctx)
        self.assertEqual(len(outputs["images"]), 2)
        self.assertEqual(outputs["mm_positions"][0], ("img_pos", "i1"))
        self.assertEqual(outputs["mm_positions"][1], ("vid_pos", "v1"))


class TestProcessPostTokens(unittest.TestCase):
    """Tests for MMProcessor._process_post_tokens."""

    def setUp(self):
        self.proc = _make_processor()

    def test_with_completion_token_ids(self):
        outputs = self.proc._make_outputs()
        request = {"completion_token_ids": [100, 200]}
        self.proc._process_post_tokens(request, outputs)
        self.assertIn(100, outputs["input_ids"])
        self.assertIn(200, outputs["input_ids"])

    def test_with_generated_token_ids(self):
        outputs = self.proc._make_outputs()
        request = {"generated_token_ids": [300, 400]}
        self.proc._process_post_tokens(request, outputs)
        self.assertIn(300, outputs["input_ids"])

    def test_no_tokens_noop(self):
        outputs = self.proc._make_outputs()
        request = {}
        self.proc._process_post_tokens(request, outputs)
        self.assertEqual(outputs["input_ids"], [])


class TestUpdateCache(unittest.TestCase):
    """Tests for MMProcessor._update_cache (hash computation + cache write)."""

    def setUp(self):
        self.proc = _make_processor()

    def test_populates_mm_hashes_without_cache(self):
        """Even without cache, mm_hashes is populated for downstream engine."""
        self.proc._cache = None
        outputs = self.proc._make_outputs()
        outputs["images"] = [np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        img = MMItem(type="image", data="raw", uuid="uuid1")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        self.assertEqual(outputs["mm_hashes"], ["uuid1"])

    @patch("fastdeploy.input.multimodal.mm_processor.MultimodalHasher.hash_features")
    def test_content_hash_fallback(self, mock_hash):
        """Without uuid, falls back to MultimodalHasher."""
        mock_hash.return_value = "computed_hash"
        self.proc._cache = None
        outputs = self.proc._make_outputs()
        outputs["images"] = [np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        img = MMItem(type="image", data="raw", uuid=None)
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        self.assertEqual(outputs["mm_hashes"], ["computed_hash"])
        mock_hash.assert_called_once()

    def test_uuid_preferred_for_hash(self):
        self.proc._cache = None
        outputs = self.proc._make_outputs()
        outputs["images"] = [np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        img = MMItem(type="image", data="raw", uuid="my_uuid")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        self.assertEqual(outputs["mm_hashes"], ["my_uuid"])

    def test_cached_items_not_re_cached(self):
        """Items with tuple data (from cache) should not be re-written to cache."""
        mock_cache = MagicMock()
        self.proc._cache = mock_cache
        outputs = self.proc._make_outputs()
        cached_pixels = np.ones((4, 3))
        outputs["images"] = [cached_pixels]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        # data is a tuple = came from cache
        img = MMItem(type="image", data=(cached_pixels, {"thw": (1, 2, 2)}), uuid="cached_uuid")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        # Hash still populated
        self.assertEqual(outputs["mm_hashes"], ["cached_uuid"])
        # But cache.put NOT called
        mock_cache.put.assert_not_called()

    def test_cache_put_called(self):
        """Newly processed items are written to cache."""
        mock_cache = MagicMock()
        self.proc._cache = mock_cache
        outputs = self.proc._make_outputs()
        pixels = np.ones((4, 3))
        outputs["images"] = [pixels]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        img = MMItem(type="image", data="raw_img", uuid="new_uuid")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        mock_cache.put.assert_called_once()
        hashes, items = mock_cache.put.call_args[0]
        self.assertEqual(hashes, ["new_uuid"])
        np.testing.assert_array_equal(items[0][0], pixels)

    def test_1d_grid_thw(self):
        """1D grid_thw correctly extracts meta."""
        mock_cache = MagicMock()
        self.proc._cache = mock_cache
        outputs = self.proc._make_outputs()
        outputs["images"] = [np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([1, 2, 2])]

        img = MMItem(type="image", data="raw", uuid="u1")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        _, items = mock_cache.put.call_args[0]
        self.assertEqual(items[0][1]["thw"], (1, 2, 2))

    def test_2d_grid_thw(self):
        """2D grid_thw extracts first row."""
        mock_cache = MagicMock()
        self.proc._cache = mock_cache
        outputs = self.proc._make_outputs()
        outputs["images"] = [np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([[2, 4, 4]])]

        img = MMItem(type="image", data="raw", uuid="u1")
        ctx = MMContext(images=[img], videos=[], mm_order=["image"])
        self.proc._update_cache(ctx, outputs)

        _, items = mock_cache.put.call_args[0]
        self.assertEqual(items[0][1]["thw"], (2, 4, 4))

    def test_no_images_noop(self):
        """Empty images list -> mm_hashes stays empty."""
        self.proc._cache = None
        outputs = self.proc._make_outputs()  # images = []
        ctx = MMContext(images=[], videos=[], mm_order=[])
        self.proc._update_cache(ctx, outputs)
        self.assertEqual(outputs["mm_hashes"], [])


class TestPackOutputs(unittest.TestCase):
    """Tests for MMProcessor._pack_outputs."""

    def setUp(self):
        self.proc = _make_processor()

    def test_with_images_vstacks(self):
        outputs = self.proc._make_outputs()
        outputs["input_ids"] = [1, 2, 3]
        outputs["token_type_ids"] = [0, 0, 0]
        outputs["images"] = [np.ones((4, 3)), np.ones((4, 3))]
        outputs["grid_thw"] = [np.array([[1, 2, 2]]), np.array([[1, 2, 2]])]
        outputs["image_type_ids"] = [1, 1]
        outputs["position_ids"] = [0, 1, 2]

        result = self.proc._pack_outputs(outputs)

        self.assertEqual(result["images"].shape, (8, 3))
        self.assertEqual(result["grid_thw"].shape, (2, 3))
        np.testing.assert_array_equal(result["input_ids"], np.array([1, 2, 3], dtype=np.int64))

    def test_without_images_sets_none(self):
        outputs = self.proc._make_outputs()
        outputs["input_ids"] = [1, 2]
        outputs["token_type_ids"] = [0, 0]
        outputs["position_ids"] = [0, 1]

        result = self.proc._pack_outputs(outputs)

        self.assertIsNone(result["images"])
        self.assertIsNone(result["grid_thw"])
        self.assertIsNone(result["image_type_ids"])

    def test_input_ids_to_int64(self):
        outputs = self.proc._make_outputs()
        outputs["input_ids"] = [1, 2, 3]
        outputs["token_type_ids"] = [0, 0, 0]
        outputs["position_ids"] = [0, 1, 2]

        result = self.proc._pack_outputs(outputs)

        self.assertEqual(result["input_ids"].dtype, np.int64)
        self.assertEqual(result["token_type_ids"].dtype, np.int64)

    def test_mm_num_token_func_set(self):
        outputs = self.proc._make_outputs()
        outputs["input_ids"] = [1]
        outputs["token_type_ids"] = [0]
        outputs["position_ids"] = [0]

        result = self.proc._pack_outputs(outputs)

        self.assertEqual(result["mm_num_token_func"], _ConcreteProcessor.mm_num_tokens)


class TestWriteBack(unittest.TestCase):
    """Tests for MMProcessor._write_back."""

    def setUp(self):
        self.proc = _make_processor()

    def test_default_writes_token_ids(self):
        request = {}
        outputs = {"input_ids": np.array([1, 2, 3], dtype=np.int64)}
        self.proc._write_back(request, outputs)
        self.assertEqual(request["prompt_token_ids"], [1, 2, 3])

    def test_sets_multimodal_inputs(self):
        request = {}
        outputs = {"input_ids": np.array([1], dtype=np.int64), "images": None}
        self.proc._write_back(request, outputs)
        self.assertIs(request["multimodal_inputs"], outputs)


class TestAddText(unittest.TestCase):
    """Tests for MMProcessor._add_text."""

    def setUp(self):
        self.proc = _make_processor()

    def test_empty_string_noop(self):
        outputs = self.proc._make_outputs()
        self.proc._add_text("", outputs)
        self.assertEqual(outputs["input_ids"], [])

    def test_string_tokenized(self):
        outputs = self.proc._make_outputs()
        self.proc._add_text("hello world", outputs)
        self.assertEqual(outputs["input_ids"], [10, 20])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 2)
        self.proc.tokenizer.tokenize.assert_called_once_with("hello world")

    def test_list_ints_direct(self):
        outputs = self.proc._make_outputs()
        self.proc._add_text([100, 200, 300], outputs)
        self.assertEqual(outputs["input_ids"], [100, 200, 300])
        self.assertEqual(outputs["token_type_ids"], [IDS_TYPE_FLAG["text"]] * 3)
        # tokenizer not called for list input
        self.proc.tokenizer.tokenize.assert_not_called()


class TestParseLimits(unittest.TestCase):
    """Tests for MMProcessor._parse_limits."""

    def setUp(self):
        self.proc = _make_processor()

    def test_none_returns_defaults(self):
        result = self.proc._parse_limits(None)
        self.assertEqual(result, _DEFAULT_MM_LIMITS)

    def test_valid_limits_merged(self):
        result = self.proc._parse_limits({"image": 5, "video": 3})
        self.assertEqual(result["image"], 5)
        self.assertEqual(result["video"], 3)
        self.assertEqual(result["audio"], 1)  # default

    def test_invalid_type_returns_defaults(self):
        result = self.proc._parse_limits("not_a_dict")
        self.assertEqual(result, _DEFAULT_MM_LIMITS)


class TestCheckMMLimits(unittest.TestCase):
    """Tests for MMProcessor._check_mm_limits."""

    def setUp(self):
        self.proc = _make_processor(limit_mm_per_prompt={"image": 2, "video": 1, "audio": 1})

    def test_within_limits_ok(self):
        self.proc._check_mm_limits({"image": ["i1", "i2"], "video": ["v1"]})

    def test_exceeds_limit_raises(self):
        with self.assertRaisesRegex(ValueError, "Too many image items"):
            self.proc._check_mm_limits({"image": ["i1", "i2", "i3"]})


class TestMakeOutputs(unittest.TestCase):
    """Tests for MMProcessor._make_outputs."""

    def setUp(self):
        self.proc = _make_processor()

    def test_has_all_expected_keys(self):
        outputs = self.proc._make_outputs()
        expected_keys = [
            "input_ids",
            "token_type_ids",
            "position_ids",
            "images",
            "grid_thw",
            "image_type_ids",
            "labels",
            "cur_position",
            "video_cnt",
            "num_input_image_tokens",
            "num_input_video_tokens",
            "mm_positions",
            "mm_hashes",
        ]
        for key in expected_keys:
            self.assertIn(key, outputs)

    def test_mm_hashes_is_list(self):
        outputs = self.proc._make_outputs()
        self.assertIsInstance(outputs["mm_hashes"], list)
        self.assertEqual(outputs["mm_hashes"], [])


class TestProcessEndToEnd(unittest.TestCase):
    """Integration tests for the full process() pipeline."""

    def setUp(self):
        self.proc = _make_processor()

    def test_full_pipeline_from_text(self):
        request = {
            "prompt": "Hello <image> world",
            "multimodal_data": {"image": [{"data": "img", "uuid": "u1"}], "mm_order": ["image"]},
        }
        self.proc.process(request)

        self.assertIn("prompt_token_ids", request)
        self.assertIn("multimodal_inputs", request)
        self.assertEqual(request["multimodal_inputs"]["mm_hashes"], ["u1"])

    def test_full_pipeline_text_only(self):
        request = {"prompt": "Hello world"}
        self.proc.process(request)

        self.assertIn("prompt_token_ids", request)
        self.assertEqual(request["multimodal_inputs"]["mm_hashes"], [])

    def test_full_pipeline_pretokenized(self):
        """PRETOKENIZED path raises NotImplementedError on base _ConcreteProcessor
        unless prompt_token_ids2outputs is overridden properly. Test via mock."""
        proc = _make_processor()
        mock_outputs = proc._make_outputs()
        mock_outputs["input_ids"] = [1, 2, 3]
        mock_outputs["token_type_ids"] = [0, 0, 0]
        mock_outputs["position_ids"] = [0, 1, 2]
        with patch.object(proc, "prompt_token_ids2outputs", return_value=mock_outputs):
            request = {
                "prompt_token_ids": [1, 2, 3],
                "multimodal_data": {"image": [{"data": "img", "uuid": "u1"}], "mm_order": ["image"]},
            }
            proc.process(request)
            self.assertIn("prompt_token_ids", request)

    def test_pipeline_with_cache_enabled(self):
        mock_cache = MagicMock()
        self.proc._cache = mock_cache

        request = {
            "prompt": "<image>",
            "multimodal_data": {"image": [{"data": "img", "uuid": "u1"}], "mm_order": ["image"]},
        }
        self.proc.process(request)

        # Cache put should be called with the hash
        mock_cache.put.assert_called_once()
        hashes, _ = mock_cache.put.call_args[0]
        self.assertEqual(hashes, ["u1"])


if __name__ == "__main__":
    unittest.main()
