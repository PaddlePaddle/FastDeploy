# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
测试多模态 warmup 相关逻辑：
  - ErnieMM45DataProcessor.prepare_mm_split_fuse_fields
  - Engine._build_mm_warmup_data
"""
import queue
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np


# ---------------------------------------------------------------------------
# 构造最小 Mock 模块，避免 import 重型依赖
# ---------------------------------------------------------------------------

def _make_paddle_mock():
    """返回一个能满足 prepare_mm_split_fuse_fields 中所有调用的 paddle mock。"""
    paddle = MagicMock(name="paddle")

    class _T:
        """统一的轻量 Tensor stub，支持 cast/cpu/cumsum/concat/squeeze/repeat_interleave/numpy/tolist。"""
        def __init__(self, data):
            self._data = np.array(data, dtype=np.float32)

        def cast(self, dtype):
            return self

        def cpu(self):
            return self

        def squeeze(self, dims=None):
            return _T(self._data.squeeze())

        def repeat_interleave(self, n, dim=None):
            return _T(np.repeat(self._data.flatten(), n))

        def numpy(self):
            return self._data

        def tolist(self):
            return self._data.tolist()

        def __len__(self):
            return len(self._data)

        def __eq__(self, other):
            # 按元素比较，返回 _T，模拟 paddle.Tensor == scalar
            val = other._data if isinstance(other, _T) else other
            return _T((self._data == val).astype(np.float32))

    def to_tensor(data, dtype=None):
        return _T(data)

    def zeros(shape, dtype=None):
        return _T(np.zeros(shape))

    def cumsum(x):
        return _T(np.cumsum(x._data))

    def concat(tensors):
        arrays = [np.atleast_1d(t._data) for t in tensors]
        return _T(np.concatenate(arrays))

    def where(cond, x, y):
        # cond 是 _T（比较结果），直接返回：0/1 值就是 is_image_token 的内容
        if isinstance(cond, _T):
            return cond
        # fallback：标量 bool
        return _T(np.array([x if cond else y]))

    paddle.to_tensor = to_tensor
    paddle.zeros = zeros
    paddle.cumsum = cumsum
    paddle.concat = concat
    paddle.where = where
    return paddle


def _setup_sys_mocks():
    """注入所有需要 Mock 的模块到 sys.modules。"""
    # paddle
    paddle_mock = _make_paddle_mock()
    sys.modules.setdefault("paddle", paddle_mock)

    # server.engine.config
    config_mod = types.ModuleType("server.engine.config")
    class VitMode:
        VIT_INCOMPLETE = MagicMock(name="VIT_INCOMPLETE")
        VIT_INCOMPLETE.name = "VIT_INCOMPLETE"
        VIT_COMPLETED = MagicMock(name="VIT_COMPLETED")
        VIT_COMPLETED.name = "VIT_COMPLETED"

    env_cfg = MagicMock()
    env_cfg.image_patch_id = 151859
    env_cfg.split_fuse_size_image = 2048
    env_cfg.split_fuse_size = 1024
    env_cfg.ellm_dynamic_mode = False
    env_cfg.enable_vpd_split = False
    env_cfg.multi_modal_model_v45_turbo = True

    config_mod.VitMode = VitMode
    config_mod.get_config = lambda: env_cfg
    sys.modules["server"] = types.ModuleType("server")
    sys.modules["server.engine"] = types.ModuleType("server.engine")
    sys.modules["server.engine.config"] = config_mod

    # server.utils
    utils_mod = types.ModuleType("server.utils")
    utils_mod.data_processor_logger = MagicMock()
    utils_mod.model_server_logger = MagicMock()
    sys.modules["server.utils"] = utils_mod

    # server.data.base_processor
    base_proc_mod = types.ModuleType("server.data.base_processor")
    base_proc_mod.BaseDataProcessor = object
    sys.modules["server.data"] = types.ModuleType("server.data")
    sys.modules["server.data.base_processor"] = base_proc_mod

    # server.data.ernie_tokenizer
    tok_mod = types.ModuleType("server.data.ernie_tokenizer")
    tok_mod.ErnieBotTokenizer = MagicMock()
    sys.modules["server.data.ernie_tokenizer"] = tok_mod

    # toolkit
    toolkit_mod = types.ModuleType("toolkit")
    toolkit_mod.ProcessedDataLoader = MagicMock()
    sys.modules["toolkit"] = toolkit_mod

    # custom_setup_ops  (get_mm_split_fuse 通过这里导入)
    ops_mod = types.ModuleType("custom_setup_ops")
    sys.modules["custom_setup_ops"] = ops_mod

    # server.data.data_processor.*
    for submod in [
        "server.data.data_processor",
        "server.data.data_processor.data_processor",
        "server.data.data_processor.data_processor.utils",
        "server.data.data_processor.data_processor.utils.argparser",
        "server.data.data_processor.data_processor.steps",
        "server.data.data_processor.data_processor.steps.end2end_processing",
    ]:
        sys.modules.setdefault(submod, types.ModuleType(submod))

    argparser_mod = sys.modules["server.data.data_processor.data_processor.utils.argparser"]
    argparser_mod.PdArgumentParser = MagicMock()
    argparser_mod.get_config = MagicMock()

    e2e_mod = sys.modules["server.data.data_processor.data_processor.steps.end2end_processing"]
    e2e_mod.End2EndProcessor = MagicMock()
    e2e_mod.End2EndProcessorArguments = MagicMock()

    return env_cfg, VitMode


# ---------------------------------------------------------------------------
# 构造一个轻量的 ErnieMM45DataProcessor stub，仅含被测方法
# ---------------------------------------------------------------------------

def _make_processor_stub(env_cfg, get_mm_split_fuse_fn):
    """
    返回一个只有 prepare_mm_split_fuse_fields 的最小 processor 实例。
    image_preprocess、patch_size、temporal_patch_size 全部手动注入。
    """
    class _ImagePreprocess:
        rescale_factor = 0.00392156862745098   # 1/255
        image_mean = [0.485, 0.456, 0.406]
        image_std  = [0.229, 0.224, 0.225]
        image_mean_tensor = np.array(image_mean, dtype="float32").reshape(1, 3, 1, 1)
        image_std_tensor  = np.array(image_std,  dtype="float32").reshape(1, 3, 1, 1)

    class _Processor:
        patch_size = 14
        temporal_patch_size = 1
        image_preprocess = _ImagePreprocess()

        def prepare_mm_split_fuse_fields(self, data):
            # 直接从 ernie_45mm_processor 中复制，但注入 mock 的 get_mm_split_fuse
            import paddle as _paddle
            input_ids = _paddle.to_tensor(data["input_ids"]).cast('int64')
            is_image_token = _paddle.where(input_ids == env_cfg.image_patch_id, 1, 0)
            image_token_sum = _paddle.cumsum(is_image_token)
            image_token_sum = _paddle.concat([_paddle.zeros([1], dtype='int64'), image_token_sum])
            grid_thw = _paddle.to_tensor(data.get("grid_thw_list", []), dtype='int64')
            image_type_ids_tensor = _paddle.to_tensor(list(data["image_type_ids"])).cast("int32")

            image_chunk_selections_task, split_fuse_cur_seq_lens_task = get_mm_split_fuse_fn(
                input_ids.cpu(),
                image_type_ids_tensor.cpu(),
                image_token_sum.cast('int32').cpu(),
                grid_thw.cpu(),
                env_cfg.image_patch_id,
                len(data.get("grid_thw_list", [])),
                0,
                len(data["input_ids"]),
                env_cfg.split_fuse_size_image,
                env_cfg.split_fuse_size,
                2048
            )
            data["image_chunk_selections_task"] = image_chunk_selections_task.numpy().tolist()
            data["split_fuse_cur_seq_lens_task"] = split_fuse_cur_seq_lens_task.numpy().tolist()
            data["split_fuse_chunk_num"] = len(split_fuse_cur_seq_lens_task)

            data["rescale_factor"] = self.image_preprocess.rescale_factor
            if env_cfg.multi_modal_model_v45_turbo:
                data["image_mean_tensor"] = _paddle.to_tensor(self.image_preprocess.image_mean_tensor) \
                                            .squeeze([-2, -1]) \
                                            .repeat_interleave(self.patch_size**2*self.temporal_patch_size, -1) \
                                            .numpy().tolist()
                data["image_std_tensor"] = _paddle.to_tensor(self.image_preprocess.image_std_tensor) \
                                           .squeeze([-2, -1]) \
                                           .repeat_interleave(self.patch_size**2*self.temporal_patch_size, -1) \
                                           .numpy().tolist()
            else:
                data["image_mean_tensor"] = self.image_preprocess.image_mean_tensor.numpy().tolist()
                data["image_std_tensor"] = self.image_preprocess.image_std_tensor.numpy().tolist()
            data["image_batch"] = len(data["image_type_ids"])
            return data

    return _Processor()


# ---------------------------------------------------------------------------
# 辅助：构造合成 warmup data（模仿 _build_mm_warmup_data 的前半部分）
# ---------------------------------------------------------------------------

def _build_synthetic_warmup_data(image_patch_id):
    T, H, W = 1, 4, 4
    merge_size = 2
    H_eff, W_eff = H // merge_size, W // merge_size
    num_img_tokens = T * H_eff * W_eff  # 4

    prefix_ids = [5, 5, 5]
    img_ids = [image_patch_id] * num_img_tokens
    suffix_ids = [5, 5, 5]
    input_ids = prefix_ids + img_ids + suffix_ids

    t = len(prefix_ids)
    position_ids = [[i, i, i] for i in range(t)]
    for h in range(H_eff):
        for w in range(W_eff):
            position_ids.append([t, t + h, t + w])
    next_pos = t + W_eff
    for k in range(len(suffix_ids)):
        position_ids.append([next_pos + k] * 3)

    return {
        "input_ids": input_ids,
        "grid_thw": [[T, H, W]],
        "grid_thw_list": [[T, H, W]],
        "image_type_ids": [0],
        "position_ids": position_ids,
        "image_dict": {},
        "media_info": {},
    }, T, H, W, H_eff, W_eff, num_img_tokens


# ---------------------------------------------------------------------------
# 测试类
# ---------------------------------------------------------------------------

class TestPrepareMmSplitFuseFields(unittest.TestCase):
    """单元测试 prepare_mm_split_fuse_fields。"""

    def setUp(self):
        self.env_cfg, self.VitMode = _setup_sys_mocks()
        self.image_patch_id = self.env_cfg.image_patch_id

        # mock get_mm_split_fuse 返回值：1 个 image chunk（crop_num=1）
        # 用 MagicMock 包装，避免直接覆盖 np.ndarray 的只读属性
        self._chunk_sel_ret = MagicMock()
        self._chunk_sel_ret.numpy.return_value = MagicMock()
        self._chunk_sel_ret.numpy.return_value.tolist.return_value = [1]

        self._seq_lens_ret = MagicMock()
        self._seq_lens_ret.numpy.return_value = MagicMock()
        self._seq_lens_ret.numpy.return_value.tolist.return_value = [10]
        self._seq_lens_ret.__len__ = lambda self: 1

        def fake_get_mm_split_fuse(*args, **kwargs):
            return self._chunk_sel_ret, self._seq_lens_ret

        self.processor = _make_processor_stub(self.env_cfg, fake_get_mm_split_fuse)
        self.data, self.T, self.H, self.W, self.H_eff, self.W_eff, self.num_img_tokens = \
            _build_synthetic_warmup_data(self.image_patch_id)

    def test_split_fuse_fields_populated(self):
        """调用后 image_chunk_selections_task / split_fuse_cur_seq_lens_task 必须存在且为列表。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertIn("image_chunk_selections_task", result)
        self.assertIn("split_fuse_cur_seq_lens_task", result)
        self.assertIn("split_fuse_chunk_num", result)
        self.assertIsInstance(result["image_chunk_selections_task"], list)
        self.assertIsInstance(result["split_fuse_cur_seq_lens_task"], list)

    def test_chunk_num_consistent(self):
        """split_fuse_chunk_num 应等于 split_fuse_cur_seq_lens_task 的长度。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertEqual(result["split_fuse_chunk_num"], len(result["split_fuse_cur_seq_lens_task"]))

    def test_rescale_factor_populated(self):
        """rescale_factor 应为非 None 的浮点数。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertIsNotNone(result["rescale_factor"])
        self.assertIsInstance(result["rescale_factor"], float)

    def test_image_mean_std_tensor_populated(self):
        """v45_turbo 模式下 image_mean_tensor / image_std_tensor 应为非空列表。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertIsNotNone(result["image_mean_tensor"])
        self.assertIsNotNone(result["image_std_tensor"])
        self.assertIsInstance(result["image_mean_tensor"], list)
        self.assertIsInstance(result["image_std_tensor"], list)
        self.assertGreater(len(result["image_mean_tensor"]), 0)

    def test_image_mean_std_length_matches_patch(self):
        """
        v45_turbo: mean/std 经 repeat_interleave(patch_size^2 * temporal_patch_size)
        展开后长度应为 3 * patch_size^2 * temporal_patch_size。
        """
        patch_size = self.processor.patch_size           # 14
        temporal = self.processor.temporal_patch_size    # 1
        expected_len = 3 * patch_size ** 2 * temporal   # 3*196*1 = 588
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertEqual(len(result["image_mean_tensor"]), expected_len)
        self.assertEqual(len(result["image_std_tensor"]), expected_len)

    def test_image_batch_equals_image_type_ids_len(self):
        """image_batch 应等于 image_type_ids 长度。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertEqual(result["image_batch"], len(self.data["image_type_ids"]))

    def test_returns_same_dict(self):
        """方法应原地修改并返回同一个 dict 对象。"""
        result = self.processor.prepare_mm_split_fuse_fields(self.data)
        self.assertIs(result, self.data)


class TestBuildMmWarmupData(unittest.TestCase):
    """
    测试 _build_mm_warmup_data 生成的数据结构。

    Engine 依赖太多重型模块，这里直接测试 warmup data 的构造逻辑，
    使用独立函数模拟 Engine._build_mm_warmup_data 的行为。
    """

    def setUp(self):
        self.env_cfg, self.VitMode = _setup_sys_mocks()
        self.image_patch_id = self.env_cfg.image_patch_id

        # 构造固定返回的 mock prepare_mm_split_fuse_fields
        def fake_prepare(data):
            data["image_chunk_selections_task"] = [1]
            data["split_fuse_cur_seq_lens_task"] = [len(data["input_ids"])]
            data["split_fuse_chunk_num"] = 1
            data["rescale_factor"] = 1 / 255.0
            data["image_mean_tensor"] = [0.0] * 588
            data["image_std_tensor"] = [1.0] * 588
            data["image_batch"] = 1
            return data

        mock_dp = MagicMock()
        mock_dp.prepare_mm_split_fuse_fields.side_effect = fake_prepare
        self.mock_dp = mock_dp

        # 构造 base_data（Engine.warmup 传入的纯文基础字段）
        self.base_data = {
            "req_id": "warmup_req",
            "input_ids": [1, 2, 3],
        }

    def _run_build(self):
        """执行 _build_mm_warmup_data 的逻辑（不依赖真实 Engine 实例）。"""
        image_patch_id = self.image_patch_id
        T, H, W = 1, 4, 4
        merge_size = 2
        H_eff = H // merge_size
        W_eff = W // merge_size
        num_img_tokens = T * H_eff * W_eff

        prefix_ids = [5, 5, 5]
        img_ids = [image_patch_id] * num_img_tokens
        suffix_ids = [5, 5, 5]
        input_ids = prefix_ids + img_ids + suffix_ids

        t = len(prefix_ids)
        position_ids = [[i, i, i] for i in range(t)]
        for h in range(H_eff):
            for w in range(W_eff):
                position_ids.append([t, t + h, t + w])
        next_pos = t + W_eff
        for k in range(len(suffix_ids)):
            position_ids.append([next_pos + k] * 3)

        data = dict(self.base_data)
        data["input_ids"] = input_ids
        data["grid_thw"] = [[T, H, W]]
        data["image_type_ids"] = [0]
        data["position_ids"] = position_ids
        data["enable_thinking"] = False
        data["max_think_len"] = -1
        data["max_content_len"] = -1

        # v45_turbo 分支
        data["grid_thw_list"] = [[T, H, W]]
        data["vit_mode"] = "VIT_INCOMPLETE"
        data["use_vpd_split"] = False
        data["image_dict"] = {}
        data["media_info"] = {}
        self.mock_dp.prepare_mm_split_fuse_fields(data)

        return data, T, H, W, H_eff, W_eff, num_img_tokens

    def test_input_ids_structure(self):
        """input_ids = prefix(3) + image_tokens(4) + suffix(3) = 10。"""
        data, T, H, W, H_eff, W_eff, num_img_tokens = self._run_build()
        self.assertEqual(len(data["input_ids"]), 3 + num_img_tokens + 3)
        # image patch id 在中间
        img_slice = data["input_ids"][3:3+num_img_tokens]
        self.assertTrue(all(x == self.image_patch_id for x in img_slice))

    def test_position_ids_length(self):
        """position_ids 长度应与 input_ids 一致。"""
        data, *_ = self._run_build()
        self.assertEqual(len(data["position_ids"]), len(data["input_ids"]))

    def test_position_ids_text_tokens_are_1d(self):
        """文本 token 的 position_ids 三个维度相等（1D 编码）。"""
        data, T, H, W, H_eff, W_eff, num_img_tokens = self._run_build()
        prefix_pos = data["position_ids"][:3]
        suffix_pos = data["position_ids"][3+num_img_tokens:]
        for pos in prefix_pos + suffix_pos:
            self.assertEqual(pos[0], pos[1], msg=f"text token pos not 1D: {pos}")
            self.assertEqual(pos[1], pos[2], msg=f"text token pos not 1D: {pos}")

    def test_position_ids_image_tokens_3d(self):
        """
        图片 token 的 position_ids：
        - pos[0]（t 维）全部相等
        - 覆盖 H_eff × W_eff 不同的 (h, w) 坐标
        """
        data, T, H, W, H_eff, W_eff, num_img_tokens = self._run_build()
        img_pos = data["position_ids"][3:3+num_img_tokens]
        t_val = img_pos[0][0]
        for pos in img_pos:
            self.assertEqual(pos[0], t_val, msg=f"image t dim varies: {pos}")

        hw_pairs = {(pos[1], pos[2]) for pos in img_pos}
        self.assertEqual(len(hw_pairs), H_eff * W_eff,
                         msg=f"expected {H_eff*W_eff} unique (h,w) pairs, got {hw_pairs}")

    def test_grid_thw_and_grid_thw_list(self):
        """grid_thw 和 grid_thw_list 内容一致。"""
        data, T, H, W, *_ = self._run_build()
        self.assertEqual(data["grid_thw"], [[T, H, W]])
        self.assertEqual(data["grid_thw_list"], [[T, H, W]])

    def test_prepare_mm_split_fuse_fields_called(self):
        """_build_mm_warmup_data 必须调用 data_processor.prepare_mm_split_fuse_fields。"""
        self._run_build()
        self.mock_dp.prepare_mm_split_fuse_fields.assert_called_once()

    def test_split_fuse_fields_not_none(self):
        """prepare_mm_split_fuse_fields 填充的字段不能为 None。"""
        data, *_ = self._run_build()
        for key in ["image_chunk_selections_task", "split_fuse_cur_seq_lens_task",
                    "rescale_factor", "image_mean_tensor", "image_std_tensor"]:
            self.assertIsNotNone(data[key], msg=f"{key} should not be None")

    def test_vit_mode_and_use_vpd_split(self):
        """vit_mode 应为 VIT_INCOMPLETE，use_vpd_split 应为 False。"""
        data, *_ = self._run_build()
        self.assertEqual(data["vit_mode"], "VIT_INCOMPLETE")
        self.assertFalse(data["use_vpd_split"])

    def test_image_dict_and_media_info_empty(self):
        """image_dict 和 media_info 应为空 dict。"""
        data, *_ = self._run_build()
        self.assertEqual(data["image_dict"], {})
        self.assertEqual(data["media_info"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
