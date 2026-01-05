"""
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
"""

import pickle
import unittest
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import zmq
from PIL import Image

from fastdeploy.input.paddleocr_vl_processor.image_processor import (
    ImageProcessor,
    smart_resize,
)
from fastdeploy.input.paddleocr_vl_processor.process import Paddleocr_VLDataProcessor
from fastdeploy.input.paddleocr_vl_processor.process_video import sample_frames

MODULE_PATH = "fastdeploy.input.paddleocr_vl_processor.process"


class TestProcessVideo(unittest.TestCase):
    def setUp(self):
        self.metadata = {"num_of_frame": 100, "fps": 25}
        self.frame_factor = 4
        self.min_frames = 8
        self.max_frames = 32

    def test_sample_with_num_frames(self):
        """测试使用num_frames参数采样（来自用户的原始测试）"""
        num_frames = 16
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            num_frames=num_frames,
            fps=0,  # 确保 fps 不>0
            metadata=self.metadata,
        )
        self.assertEqual(len(indices), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 93)
        np.testing.assert_array_equal(indices, np.arange(0, 100, 100 / 16).astype(np.int32))

    def test_error_num_frames_exceeds_total(self):
        """测试 num_frames 超过总帧数的异常（来自用户的原始测试）"""
        with self.assertRaises(ValueError) as context:
            sample_frames(
                frame_factor=self.frame_factor,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
                num_frames=200,  # 超过总帧数100
                fps=0,
                metadata=self.metadata,
            )
        self.assertIn("exceeds", str(context.exception))

    def test_error_mutual_exclusion(self):
        """新增：测试 num_frames 和 fps 互斥"""
        with self.assertRaises(ValueError) as context:
            sample_frames(
                frame_factor=self.frame_factor,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
                num_frames=16,  # > 0
                fps=10,  # > 0
                metadata=self.metadata,
            )
        self.assertIn("mutually exclusive", str(context.exception))

    def test_error_fps_without_metadata(self):
        """新增：测试 fps > 0 但 metadata 为 None"""
        with self.assertRaises(TypeError) as context:
            sample_frames(
                frame_factor=self.frame_factor,
                min_frames=self.min_frames,
                max_frames=self.max_frames,
                num_frames=0,
                fps=10,
                metadata=None,  # 缺失
            )
        # 验证是预期的 TypeError
        self.assertIn("'NoneType' object is not subscriptable", str(context.exception))

    def test_num_frames_rounding(self):
        """新增：测试 num_frames 向 frame_factor 舍入"""
        num_frames = 17  # 不是 4 的倍数
        # 逻辑: round(17 / 4) * 4 = round(4.25) * 4 = 4 * 4 = 16
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            num_frames=num_frames,
            fps=0,
            metadata=self.metadata,
        )
        # 应舍入到 16
        self.assertEqual(len(indices), 16)

    def test_sample_with_fps_basic(self):
        """新增：测试使用 fps 采样（基本路径，被 max_frames 限制）"""
        # 逻辑: num_frames_calc = 100 / 25 * 10 = 40
        #      num_frames_clamped = min(max(40, 8), 32) = 32
        #      num_frames_factored = floor(32 / 4) * 4 = 32
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            num_frames=0,
            fps=10,
            metadata=self.metadata,
        )
        # 应被 max_frames=32 限制
        self.assertEqual(len(indices), 32)
        self.assertEqual(indices[-1], 96)

    def test_sample_with_fps_hits_min_frames(self):
        """新增：测试使用 fps 采样（被 min_frames 限制）"""
        # 逻辑: num_frames_calc = 100 / 25 * 1 = 4
        #      num_frames_clamped = min(max(4, 8), 32) = 8
        #      num_frames_factored = floor(8 / 4) * 4 = 8
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            num_frames=0,
            fps=1,
            metadata=self.metadata,
        )
        # 应被 min_frames=8 限制
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[-1], 87)

    def test_sample_with_fps_hits_total_frames(self):
        """新增：测试使用 fps 采样（被 total_num_frames 限制）"""
        local_max_frames = 200

        # 逻辑: num_frames_calc = 100 / 25 * 50 = 200
        #      num_frames_clamped = min(min(max(200, 8), 200), 100) = 100
        #      num_frames_factored = floor(100 / 4) * 4 = 100
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=local_max_frames,
            num_frames=0,
            fps=50,
            metadata=self.metadata,
        )
        # 应被 total_num_frames=100 限制
        self.assertEqual(len(indices), 100)
        self.assertEqual(indices[-1], 99)  # 采样所有帧

    def test_no_sampling(self):
        """新增：测试不采样（fps=0, num_frames=0）"""
        indices = sample_frames(
            frame_factor=self.frame_factor,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
            num_frames=0,
            fps=0,
            metadata=self.metadata,
        )
        # 应返回所有帧
        self.assertEqual(len(indices), self.metadata["num_of_frame"])
        self.assertEqual(len(indices), 100)
        self.assertEqual(indices[-1], 99)
        np.testing.assert_array_equal(indices, np.arange(0, 100).astype(np.int32))


class Test_Paddleocr_VLDataProcessor(unittest.TestCase):
    """
    针对 process.py 中 Paddleocr_VLDataProcessor 类的单元测试。
    """

    def setUp(self):

        # 1. 手动启动 Patcher
        patcher_zmq_context = patch(f"{MODULE_PATH}.zmq.Context")

        self.mock_zmq_context_constructor = patcher_zmq_context.start()

        # 2. 创建模拟对象
        self.mock_tokenizer = MagicMock()
        self.mock_image_processor = MagicMock()
        self.mock_zmq_context = MagicMock()
        self.mock_zmq_socket = MagicMock()

        # 3. 配置 from_pretrained 和 zmq
        self.mock_zmq_context_constructor.return_value = self.mock_zmq_context
        self.mock_zmq_context.socket.return_value = self.mock_zmq_socket
        self.mock_image_processor.from_pretrained = MagicMock(return_value=self.mock_image_processor)

        # 4. 配置模拟对象的属性和方法
        self._configure_mocks()

        # 5. 实例化 DataProcessor (默认不启用 cache)
        with patch(
            "fastdeploy.input.paddleocr_vl_processor.process.ImageProcessor",
            self.mock_image_processor,
        ):
            with patch("os.path.exists", return_value=True):
                self.processor = Paddleocr_VLDataProcessor(
                    tokenizer=self.mock_tokenizer, image_preprocessor_name="test_model"
                )
        self._configure_processor_ids()

        # 6. 准备测试用的虚拟数据
        self.dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        self.dummy_video_frames = np.uint8(np.random.rand(16, 224, 224, 3) * 255)
        self.dummy_video_data = "path/to/dummy_video.mp4"
        self.dummy_processed_image_cache = (
            np.random.rand(64, 3, 14, 14).astype(np.float32),
            {"thw": (1, 8, 8), "fps": 0},
        )
        self.dummy_processed_video_cache = (
            np.random.rand(256, 3, 14, 14).astype(np.float32),
            {"thw": (4, 8, 8), "fps": 30},
        )

    def _configure_mocks(self):
        def mock_convert_tokens_to_ids(tokens):
            if tokens == "<|IMAGE_PLACEHOLDER|>":
                return 100
            if tokens == "<|video_pad|>":
                return 101
            if tokens == "<|IMAGE_START|>":
                return 102
            if isinstance(tokens, list):
                if tokens == ["Hello", "world"]:
                    return [983, 984]
                if tokens == ["Prompt", "text"]:
                    return [606, 511]
                if tokens == ["Prompt", "", "text"]:
                    return [606, 511]  # 模拟 "Prompt  text".split()
                return [hash(t) % 1000 for t in tokens]
            return hash(tokens) % 1000

        self.mock_tokenizer.convert_tokens_to_ids.side_effect = mock_convert_tokens_to_ids
        self.mock_tokenizer.tokenize.side_effect = lambda s: s.split()
        self.mock_tokenizer.ignored_index = -100
        self.mock_tokenizer.chat_template = "dummy_template_string"

        self.mock_image_processor.merge_size = 2
        self.mock_image_processor.temporal_patch_size = 1

    def _configure_processor_ids(self):
        self.processor.image_token_id = 100
        self.processor.video_token_id = 101
        self.processor.image_patch_id = 100
        self.processor.vision_start_id = 102

    def _get_init_outputs(self):
        return {
            "input_ids": [],
            "token_type_ids": [],
            "position_ids": [],
            "images": [],
            "grid_thw": [],
            "image_type_ids": [],
            "labels": [],
            "cur_position": 0,
            "video_cnt": 0,
            "num_input_image_tokens": 0,
            "num_input_video_tokens": 0,
            "fps": [],
            "mm_positions": [],
            "mm_hashes": [],
            "vit_seqlen": [],
            "vit_position_ids": [],
        }

    def test_compute_text_positions(self):
        """测试 _compute_text_positions 纯函数"""
        pos_ids = self.processor._compute_text_positions(start_pos=5, num_tokens=3)
        expected = np.array([[5, 6, 7], [5, 6, 7], [5, 6, 7]])
        np.testing.assert_array_equal(pos_ids, expected)

    def test_compute_vision_positions(self):
        """测试 _compute_vision_positions 纯函数"""
        pos_ids = self.processor._compute_vision_positions(start_pos=10, t=2, h=4, w=4, second_per_grid_t=1.0)
        self.assertEqual(pos_ids.shape, (3, 8))
        expected_t = np.array([0, 0, 0, 0, 2, 2, 2, 2])
        expected_h = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        expected_w = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        expected = np.stack([expected_t, expected_h, expected_w]) + 10
        np.testing.assert_array_equal(pos_ids, expected)

    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_text(self):
        """测试 _add_text 辅助函数"""
        outputs = self._get_init_outputs()
        self.mock_tokenizer.tokenize.return_value = ["Hello", "world"]
        self.mock_tokenizer.convert_tokens_to_ids.side_effect = None
        self.mock_tokenizer.convert_tokens_to_ids.return_value = [10, 11]

        self.processor._add_text("Hello world", outputs)

        self.assertEqual(outputs["input_ids"], [10, 11])
        self.assertEqual(outputs["token_type_ids"], [0, 0])
        self.assertEqual(outputs["cur_position"], 2)

    @patch(f"{MODULE_PATH}.MultimodalHasher.hash_features", return_value="dummy_hash_123")
    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_image_autohash(self, mock_hasher):
        """测试 _add_image 辅助函数 (自动哈希)"""
        outputs = self._get_init_outputs()
        outputs["cur_position"] = 5

        num_patches_hw = 8 * 8
        num_tokens = 16
        mock_preprocess_return = {
            "pixel_values": np.random.rand(num_patches_hw, 3, 14, 14),
            "grid_thw": np.array([1, 8, 8]),
        }
        self.mock_image_processor.preprocess.return_value = mock_preprocess_return

        self.processor._add_image(self.dummy_image, outputs, uuid=None)

        self.assertEqual(len(outputs["input_ids"]), num_tokens)
        self.assertEqual(outputs["num_input_image_tokens"], num_tokens)
        mock_hasher.assert_called_once_with(mock_preprocess_return["pixel_values"])
        self.assertEqual(outputs["mm_hashes"][0], "dummy_hash_123")
        self.assertEqual(outputs["cur_position"], 9)

    @patch(f"{MODULE_PATH}.MultimodalHasher.hash_features")
    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_video_with_uuid(self, mock_hasher):
        """测试 _add_video 辅助函数 (使用 uuid)"""
        outputs = self._get_init_outputs()
        outputs["cur_position"] = 10
        meta = {"fps": 30}

        num_patches_total = 256
        num_tokens = 64

        mock_preprocess_return = {
            "pixel_values": np.random.rand(num_patches_total, 3, 14, 14),
            "image_grid_thw": np.array([4, 8, 8]),
        }
        self.mock_image_processor.preprocess.return_value = mock_preprocess_return

        self.processor._add_video(self.dummy_video_frames, meta, outputs, uuid="custom_vid_uuid")

        self.assertEqual(len(outputs["input_ids"]), num_tokens)
        self.assertEqual(outputs["token_type_ids"], [2] * num_tokens)
        mock_hasher.assert_not_called()
        self.assertEqual(outputs["mm_hashes"][0], "custom_vid_uuid")
        self.assertEqual(outputs["image_type_ids"], [1, 1, 1, 1])

    @patch.object(Paddleocr_VLDataProcessor, "_add_text", MagicMock())
    @patch.object(Paddleocr_VLDataProcessor, "_add_image", MagicMock())
    @patch.object(Paddleocr_VLDataProcessor, "_add_video", MagicMock())
    @patch.object(Paddleocr_VLDataProcessor, "_load_and_process_video")
    def test_text2ids_parsing(self, mock_load_video):
        """测试 text2ids 的解析和分支逻辑"""
        mock_load_video.return_value = (self.dummy_video_frames, {"fps": 30})
        text = "Text1 <|IMAGE_PLACEHOLDER|> Text2 <|video_pad|> Text3"
        images = [self.dummy_image]
        videos = [self.dummy_video_data]
        image_uuid = ["img_uuid_1"]
        video_uuid = ["vid_uuid_1"]

        outputs = self.processor.text2ids(text, images, videos, image_uuid, video_uuid)

        self.processor._add_text.assert_any_call("Text1 ", outputs)
        self.processor._add_image.assert_called_once_with(self.dummy_image, outputs, "img_uuid_1")
        self.processor._add_video.assert_called_once_with(self.dummy_video_frames, {"fps": 30}, outputs, "vid_uuid_1")

    @patch(f"{MODULE_PATH}.parse_chat_messages")
    @patch.object(Paddleocr_VLDataProcessor, "text2ids", return_value="final_output")
    def test_request2ids(self, mock_text2ids, mock_parse_chat):
        """测试 request2ids 的 chat 模板逻辑"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "data": self.dummy_image, "uuid": "img1"},
                ],
            }
        ]
        request = {"messages": messages, "add_generation_prompt": True}
        mock_parse_chat.return_value = messages
        parsed_prompt = "User: Hello <|IMAGE_PLACEHOLDER|> Assistant:"
        self.mock_tokenizer.apply_chat_template.return_value = parsed_prompt

        result = self.processor.request2ids(request)

        self.mock_tokenizer.apply_chat_template.assert_called_once()
        mock_text2ids.assert_called_once_with(parsed_prompt, [self.dummy_image], [], ["img1"], [])
        self.assertEqual(result, "final_output")

    @patch(f"{MODULE_PATH}.sample_frames")
    @patch(f"{MODULE_PATH}.read_video_decord")
    def test_load_and_process_video(self, mock_read_video, mock_sample_frames):
        """测试 _load_and_process_video 的帧采样逻辑"""
        mock_reader = MagicMock()
        mock_reader.__getitem__.return_value.asnumpy.return_value = np.random.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        mock_meta = {"num_of_frame": 100, "duration": 10.0, "fps": 10.0}
        mock_read_video.return_value = (mock_reader, mock_meta, None)
        mock_sample_frames.return_value = [0, 10, 20, 30, 40]
        self.processor.fps = 1

        frames, meta = self.processor._load_and_process_video("dummy_url", {"min_frames": 2, "max_frames": 10})

        mock_sample_frames.assert_called_once_with(
            frame_factor=ANY,
            min_frames=2,
            max_frames=10,
            metadata=mock_meta,
            fps=self.processor.fps,
            num_frames=self.processor.target_frames,
        )
        self.assertEqual(frames.shape, (5, 100, 100, 3))
        self.assertEqual(meta["fps"], 1)

    def test_init_with_external_tokenizer(self):
        """新增：测试使用外部传入的 tokenizer 初始化"""

        external_tokenizer = MagicMock()
        with patch(
            "fastdeploy.input.paddleocr_vl_processor.process.ImageProcessor",
            MagicMock(),
        ):
            with patch("os.path.exists", return_value=True):
                processor = Paddleocr_VLDataProcessor(
                    tokenizer=external_tokenizer, image_preprocessor_name="test_model"
                )

        self.assertIs(processor.tokenizer, external_tokenizer)

    def test_add_text_empty(self):
        """新增：测试 _add_text 传入空字符串"""
        outputs = self._get_init_outputs()
        self.processor._add_text("", outputs)
        self.assertEqual(outputs["input_ids"], [])
        self.assertEqual(outputs["cur_position"], 0)

    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0})
    def test_add_text_pre_tokenized(self):
        """新增：测试 _add_text 传入已 tokenized 的 IDs"""
        outputs = self._get_init_outputs()
        token_ids = [10, 11, 12]
        self.processor._add_text(token_ids, outputs)

        self.mock_tokenizer.tokenize.assert_not_called()
        self.assertEqual(outputs["input_ids"], [10, 11, 12])
        self.assertEqual(outputs["token_type_ids"], [0, 0, 0])
        self.assertEqual(outputs["cur_position"], 3)

    @patch(f"{MODULE_PATH}.MultimodalHasher.hash_features", return_value="dummy_hash_456")
    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_video_no_uuid(self, mock_hasher):
        """新增：测试 _add_video 在 uuid 为 None 时自动哈希"""
        outputs = self._get_init_outputs()
        meta = {"fps": 30}
        mock_preprocess_return = {
            "pixel_values": np.random.rand(256, 3, 14, 14),
            "image_grid_thw": np.array([4, 8, 8]),
        }
        self.mock_image_processor.preprocess.return_value = mock_preprocess_return

        self.processor._add_video(self.dummy_video_frames, meta, outputs, uuid=None)

        mock_hasher.assert_called_once_with(mock_preprocess_return["pixel_values"])
        self.assertEqual(outputs["mm_hashes"][0], "dummy_hash_456")

    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_processed_image(self):
        """新增：测试 _add_processed_image 处理缓存数据"""
        outputs = self._get_init_outputs()
        outputs["cur_position"] = 3

        self.processor._add_processed_image(self.dummy_processed_image_cache, outputs, "cached_img_uuid")

        num_tokens = 16
        self.assertEqual(len(outputs["input_ids"]), num_tokens)
        self.assertEqual(outputs["input_ids"][0], self.processor.image_patch_id)

        np.testing.assert_array_equal(outputs["images"][0], self.dummy_processed_image_cache[0])

        self.assertEqual(outputs["mm_hashes"][0], "cached_img_uuid")
        self.assertEqual(outputs["cur_position"], 7)

    @patch(f"{MODULE_PATH}.IDS_TYPE_FLAG", {"text": 0, "image": 1, "video": 2})
    def test_add_processed_video(self):
        """新增：测试 _add_processed_video 处理缓存数据"""
        outputs = self._get_init_outputs()
        outputs["cur_position"] = 5

        self.processor._add_processed_video(self.dummy_processed_video_cache, outputs, "cached_vid_uuid")

        num_tokens = 64
        t, h, w = self.dummy_processed_video_cache[1]["thw"]

        self.assertEqual(len(outputs["input_ids"]), num_tokens)
        self.assertEqual(outputs["token_type_ids"], [2] * num_tokens)

        np.testing.assert_array_equal(outputs["images"][0], self.dummy_processed_video_cache[0])

        self.assertEqual(outputs["mm_hashes"][0], "cached_vid_uuid")
        self.assertEqual(outputs["image_type_ids"], [1] * t)
        self.assertGreater(outputs["cur_position"], 5)

    def test_text2ids_with_processed_data(self):
        """新增：测试 text2ids 调用 _add_processed_image 和 _add_processed_video"""
        with (
            patch.object(self.processor, "_add_processed_image") as mock_add_proc_img,
            patch.object(self.processor, "_add_processed_video") as mock_add_proc_vid,
        ):

            text = "<|IMAGE_PLACEHOLDER|><|video_pad|>"
            images = [self.dummy_processed_image_cache]
            videos = [self.dummy_processed_video_cache]
            image_uuid = ["img1"]
            video_uuid = ["vid1"]

            self.processor.text2ids(text, images, videos, image_uuid, video_uuid)

            mock_add_proc_img.assert_called_once_with(self.dummy_processed_image_cache, ANY, "img1")
            mock_add_proc_vid.assert_called_once_with(self.dummy_processed_video_cache, ANY, "vid1")

    @patch(f"{MODULE_PATH}.sample_frames")
    @patch(f"{MODULE_PATH}.read_video_decord")
    def test_load_and_process_video_no_sampling(self, mock_read_video, mock_sample_frames):
        """新增：测试 _load_and_process_video 不采样（fps=-1）"""
        mock_reader = MagicMock()
        mock_reader.__getitem__.return_value.asnumpy.return_value = np.random.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        mock_meta = {"num_of_frame": 10, "duration": 1.0, "fps": 10.0}
        mock_read_video.return_value = (mock_reader, mock_meta, None)

        self.processor.fps = -1
        self.processor.target_frames = -1

        frames, meta = self.processor._load_and_process_video("dummy_url", {})

        mock_sample_frames.assert_not_called()
        self.assertEqual(frames.shape, (10, 100, 100, 3))
        self.assertEqual(meta["num_of_frame"], 10)

    def test_get_processor_cache(self):
        """新增：测试 get_processor_cache (zmq)"""
        hashes = ["hash1", "hash2"]
        expected_items = ["item1", "item2"]
        mock_resp = pickle.dumps(expected_items)
        self.mock_zmq_socket.recv_multipart.return_value = (b"", mock_resp)

        items = self.processor.get_processor_cache(self.mock_zmq_socket, hashes)

        self.mock_zmq_socket.send_multipart.assert_called_once_with([b"", pickle.dumps(hashes)])
        self.assertEqual(items, expected_items)

    def test_update_processor_cache(self):
        """新增：测试 update_processor_cache (zmq)"""
        hashes = ["hash1"]
        items = ["item1"]

        self.processor.update_processor_cache(self.mock_zmq_socket, hashes, items)

        expected_req = pickle.dumps((hashes, items))
        self.mock_zmq_socket.send_multipart.assert_called_once_with([b"", expected_req])

    def test_apply_chat_template(self):
        """新增：测试 apply_chat_template 核心逻辑"""
        request = {"messages": ["msg1"], "add_generation_prompt": True, "request_id": "req123"}
        self.mock_tokenizer.apply_chat_template.return_value = "Prompt <|IMAGE_PLACEHOLDER|> text"
        self.mock_tokenizer.tokenize.return_value = ["Prompt", "text"]

        self.mock_tokenizer.convert_tokens_to_ids.side_effect = None
        self.mock_tokenizer.convert_tokens_to_ids.return_value = [10, 11]

        token_ids = self.processor.apply_chat_template(request)

        self.assertEqual(token_ids, [10, 11])
        self.assertEqual(request["text_after_process"], "Prompt <|IMAGE_PLACEHOLDER|> text")

        self.mock_tokenizer.tokenize.assert_called_with("Prompt  text")

    def test_apply_chat_template_raises_error(self):
        """新增：测试 apply_chat_template 在模板不存在时引发 ValueError"""
        self.mock_tokenizer.chat_template = None
        with self.assertRaises(ValueError) as context:
            self.processor.apply_chat_template({"messages": []})
        self.assertIn("does not support chat_template", str(context.exception))

    @patch(f"{MODULE_PATH}.parse_chat_messages")
    def test_request2ids_cache_miss_raises_error(self, mock_parse_chat):
        """新增：测试 request2ids 在缓存关闭时缺少数据引发 ValueError"""
        messages = [{"role": "user", "content": [{"type": "image", "uuid": "img1"}]}]
        request = {"messages": messages}

        mock_parse_chat.return_value = messages

        with self.assertRaises(ValueError) as context:
            self.processor.request2ids(request)

        self.assertIn("Missing items cannot be retrieved without processor cache.", str(context.exception))

    @patch(f"{MODULE_PATH}.Paddleocr_VLDataProcessor.get_processor_cache")
    @patch(f"{MODULE_PATH}.Paddleocr_VLDataProcessor.update_processor_cache")
    @patch(f"{MODULE_PATH}.Paddleocr_VLDataProcessor.text2ids")
    @patch(f"{MODULE_PATH}.parse_chat_messages")
    def test_request2ids_cache_hit_and_update(self, mock_parse_chat, mock_text2ids, mock_update_cache, mock_get_cache):
        """新增：测试 request2ids 缓存命中和缓存更新"""
        mock_tokenizer = MagicMock()
        with patch(
            "fastdeploy.input.paddleocr_vl_processor.process.ImageProcessor",
            MagicMock(),
        ):
            self.processor = Paddleocr_VLDataProcessor(tokenizer=mock_tokenizer, enable_processor_cache=True)
        self._configure_processor_ids()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "uuid": "img_cache_hit"},
                    {"type": "image", "data": self.dummy_image, "uuid": "img_to_update"},
                ],
            }
        ]
        request = {"messages": messages}

        mock_parse_chat.return_value = messages
        mock_get_cache.return_value = [self.dummy_processed_image_cache]

        mock_text2ids_output = {
            "grid_thw": [(1, 8, 8), (1, 8, 8)],
            "fps": [0, 0],
            "mm_hashes": ["img_cache_hit", "img_to_update"],
            "images": [self.dummy_processed_image_cache[0], self.dummy_processed_image_cache[0]],
        }
        mock_text2ids.return_value = mock_text2ids_output
        self.mock_tokenizer.apply_chat_template.return_value = "<|IMAGE_PLACEHOLDER|><|IMAGE_PLACEHOLDER|>"

        self.processor.request2ids(request)

        self.mock_zmq_context.socket.assert_called_with(zmq.DEALER)
        mock_get_cache.assert_called_once_with(self.mock_zmq_socket, ["img_cache_hit"])

        parsed_images = mock_text2ids.call_args[0][1]
        self.assertIs(parsed_images[0], self.dummy_processed_image_cache)
        self.assertIs(parsed_images[1], self.dummy_image)

        expected_hash_to_cache = ["img_to_update"]
        expected_item_to_cache = (self.dummy_processed_image_cache[0], {"thw": (1, 8, 8), "fps": 0})
        mock_update_cache.assert_called_once()
        self.assertEqual(mock_update_cache.call_args[0][1], expected_hash_to_cache)
        self.assertEqual(mock_update_cache.call_args[0][2][0][1], expected_item_to_cache[1])
        np.testing.assert_array_equal(mock_update_cache.call_args[0][2][0][0], expected_item_to_cache[0])

    @patch(f"{MODULE_PATH}.Paddleocr_VLDataProcessor.text2ids")
    @patch(f"{MODULE_PATH}.parse_chat_messages")
    def test_request2ids_unsupported_type(self, mock_parse_chat, mock_text2ids):
        """新增：测试 request2ids 静默忽略不支持的类型"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}, {"type": "audio", "data": "...", "uuid": "audio1"}],
            }
        ]
        request = {"messages": messages}

        mock_parse_chat.return_value = messages
        self.mock_tokenizer.apply_chat_template.return_value = "User: Hello "

        self.processor.request2ids(request)

        mock_text2ids.assert_called_once()
        call_args = mock_text2ids.call_args[0]
        self.assertEqual(call_args[1], [])  # images
        self.assertEqual(call_args[2], [])  # videos
        self.assertEqual(call_args[3], [])  # image_uuid
        self.assertEqual(call_args[4], [])  # video_uuid


class TestPaddleOCR_VL_ImageProcessor(unittest.TestCase):
    def setUp(self):
        # 初始化默认参数
        self.default_params = {
            "do_resize": True,
            "resample": 3,
            "do_rescale": True,
            "rescale_factor": 1 / 255,
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "do_convert_rgb": True,
            "min_pixels": 28 * 28 * 130,
            "max_pixels": 28 * 28 * 1280,
            "patch_size": 14,
            "temporal_patch_size": 1,
            "merge_size": 2,
        }

        # 创建测试图像
        self.test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    def test_initialization(self):
        """测试初始化参数是否正确设置"""
        processor = ImageProcessor(**self.default_params)

        for param, value in self.default_params.items():
            self.assertEqual(getattr(processor, param), value)

    def test_smart_resize(self):
        """测试智能调整图像大小功能"""
        # 测试正常尺寸调整
        h, w = smart_resize(224, 224, factor=28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

        # 测试小尺寸调整
        h, w = smart_resize(20, 20, factor=28)
        self.assertGreaterEqual(h, 28)
        self.assertGreaterEqual(w, 28)

        # 测试超大尺寸调整
        h, w = smart_resize(2000, 2000, factor=28)
        self.assertLess(h * w, 28 * 28 * 1280)

    def test_preprocess_single_image(self):
        """测试单张图像预处理流程"""
        processor = ImageProcessor(**self.default_params)

        # 测试正常预处理
        result = processor.preprocess(self.test_image)
        self.assertIn("pixel_values", result)
        self.assertIn("grid_thw", result)
        self.assertEqual(result["pixel_values"].ndim, 4)  # [N, C, H, W]

        # 测试关闭某些预处理步骤
        result = processor.preprocess(self.test_image, do_resize=False, do_normalize=False)
        self.assertIn("pixel_values", result)

    def test_preprocess_batch_images(self):
        """测试批量图像预处理"""
        processor = ImageProcessor(**self.default_params)
        batch_images = [self.test_image, self.test_image]

        result = processor.preprocess(batch_images)
        expected_shape = 1152
        self.assertEqual(result["pixel_values"].shape[0], expected_shape)

    def test_invalid_input(self):
        """测试无效输入处理"""
        processor = ImageProcessor(**self.default_params)

        # 测试无效图像
        with self.assertRaises(ValueError):
            processor.preprocess("invalid_image")

        # 测试视频输入(暂不支持)
        with self.assertRaises(NotImplementedError):
            processor.preprocess(self.test_image, videos=["video"])

    def test_from_pretrained(self):
        """测试从预训练模型加载配置"""
        with patch("builtins.open", unittest.mock.mock_open(read_data='{"do_resize": false}')) as mock_file:
            processor = ImageProcessor.from_pretrained("dummy_path")
            self.assertFalse(processor.do_resize)
            mock_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
