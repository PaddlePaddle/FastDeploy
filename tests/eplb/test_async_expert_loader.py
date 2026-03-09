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

import contextlib
import ctypes
import json
from unittest.mock import MagicMock, patch

import numpy as np
import paddle
import pytest

from fastdeploy.config import EPLBConfig
from fastdeploy.eplb.async_expert_loader import (
    AsyncEPLoader,
    create_mmap,
    load_ep_checkpoint,
    load_model_weights_process,
    load_tensor_from_shm_mem,
    save_tensor_to_shm_mem,
)


def _eplb_config(**overrides):
    defaults = {
        "redundant_expert_async_load_model_shmem_size_gb": 1,
        "model_use_safetensors": False,
        "moe_quant_type": "",
    }
    defaults.update(overrides)
    return EPLBConfig(defaults)


def _make_loader(safetensors=False, **kw):
    cfg = _eplb_config(model_use_safetensors=safetensors)
    logger = MagicMock()
    defaults = dict(
        model_dir="/fake/model",
        eplb_config=cfg,
        rank=0,
        expert_per_rank=2,
        moe_layer_start_index=1,
        moe_quant_type="",
        logger=logger,
    )
    defaults.update(kw)
    return AsyncEPLoader(**defaults)


def _shm_buffer(data_bytes):
    buf = (ctypes.c_byte * len(data_bytes))(*data_bytes)
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_int8))


# ── Shared-memory I/O ──────────────────────────────────────────────────────
class TestSharedMemoryIO:
    """save_tensor_to_shm_mem + load_tensor_from_shm_mem."""

    # save
    def test_save_single(self, tmp_path):
        fp = tmp_path / "shm"
        fp.write_bytes(b"\x00" * 4096)
        t = paddle.ones([2, 3], dtype="float32")
        infos = save_tensor_to_shm_mem([("w1", t)], str(fp))
        name, offset, size, shape, dtype = infos[0]
        assert (name, offset, size) == ("w1", 0, 24)
        assert list(shape) == [2, 3] and dtype == paddle.float32

    def test_save_multiple_offsets(self, tmp_path):
        fp = tmp_path / "shm"
        fp.write_bytes(b"\x00" * 8192)
        t1 = paddle.ones([4], dtype="float32")
        t2 = paddle.zeros([8], dtype="float32")
        infos = save_tensor_to_shm_mem([("w1", t1), ("w2", t2)], str(fp))
        assert infos[0][1] == 0 and infos[1][1] == 16

    def test_save_file_not_exist(self):
        with pytest.raises(OSError):
            save_tensor_to_shm_mem([], "/nonexistent/path")

    def test_save_overflow(self, tmp_path):
        fp = tmp_path / "tiny"
        fp.write_bytes(b"\x00" * 4)
        with pytest.raises(IOError):
            save_tensor_to_shm_mem([("big", paddle.ones([100], dtype="float32"))], str(fp))

    def test_save_with_logger(self, tmp_path):
        fp = tmp_path / "shm"
        fp.write_bytes(b"\x00" * 4096)
        logger = MagicMock()
        save_tensor_to_shm_mem([("w", paddle.ones([2], dtype="float32"))], str(fp), logger=logger)
        logger.info.assert_called_once()

    # load
    @pytest.mark.parametrize(
        "np_dtype,pd_dtype,vals",
        [
            (np.float32, paddle.float32, [1.0, 2.0, 3.0]),
            (np.uint8, paddle.uint8, [0, 128, 255]),
            (np.int8, paddle.int8, [-1, 0, 127]),
            (np.int32, paddle.int32, [10, 20, 30]),
        ],
    )
    def test_load_numeric_dtypes(self, np_dtype, pd_dtype, vals):
        arr = np.array(vals, dtype=np_dtype)
        raw = arr.tobytes()
        result = load_tensor_from_shm_mem([("w", 0, len(raw), [len(vals)], pd_dtype)], _shm_buffer(raw))
        np.testing.assert_array_equal(result[0][1].numpy(), arr)

    def test_load_bfloat16(self):
        arr = np.array([0x3F80, 0x4000], dtype=np.uint16)
        raw = arr.tobytes()
        result = load_tensor_from_shm_mem([("w", 0, len(raw), [2], paddle.bfloat16)], _shm_buffer(raw))
        assert list(result[0][1].shape) == [2]

    def test_load_float8_e4m3fn(self):
        arr = np.array([0x38, 0x40], dtype=np.uint8)
        raw = arr.tobytes()
        result = load_tensor_from_shm_mem([("w", 0, len(raw), [2], paddle.float8_e4m3fn)], _shm_buffer(raw))
        assert list(result[0][1].shape) == [2]

    def test_load_unsupported_dtype(self):
        with pytest.raises(TypeError):
            load_tensor_from_shm_mem([("w", 0, 8, [2], paddle.complex64)], _shm_buffer(b"\x00" * 8))

    def test_load_multiple_at_offsets(self):
        a1 = np.array([1.0, 2.0], dtype=np.float32)
        a2 = np.array([3, 4, 5], dtype=np.int32)
        raw = a1.tobytes() + a2.tobytes()
        infos = [
            ("w1", 0, len(a1.tobytes()), [2], paddle.float32),
            ("w2", len(a1.tobytes()), len(a2.tobytes()), [3], paddle.int32),
        ]
        result = load_tensor_from_shm_mem(infos, _shm_buffer(raw))
        np.testing.assert_allclose(result[0][1].numpy(), a1)
        np.testing.assert_array_equal(result[1][1].numpy(), a2)

    def test_load_with_logger(self):
        arr = np.array([1.0], dtype=np.float32)
        logger = MagicMock()
        load_tensor_from_shm_mem([("w", 0, 4, [1], paddle.float32)], _shm_buffer(arr.tobytes()), logger)
        logger.info.assert_called_once()


# ── AsyncEPLoader + helpers ────────────────────────────────────────────────
class TestAsyncEPLoader:
    """AsyncEPLoader class, load_ep_checkpoint, load_weight_bf16, load_safetensor_fp8."""

    # load_ep_checkpoint
    def test_checkpoint_missing(self):
        assert load_ep_checkpoint("/nonexistent") == {}

    def test_checkpoint_parses(self, tmp_path):
        data = {"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(data))
        result = load_ep_checkpoint(str(tmp_path))
        assert len(result) == 2

    # init / reset
    def test_init_and_reset(self):
        loader = _make_loader()
        assert loader.model_path == "/fake/model"
        loader.old_model_ep_rank_to_expert_id_list = np.array([[1, 2]])
        loader.cached_weights = [("x", "y")]
        loader.reset()
        assert loader.old_model_ep_rank_to_expert_id_list is None
        assert loader.cached_weights == []

    # load_experts_weight_from_disk
    def test_load_bf16_path(self):
        loader = _make_loader(safetensors=False)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        with patch.object(loader, "load_weight_bf16_from_disk", return_value=(True, "ok")) as m:
            ok, _ = loader.load_experts_weight_from_disk()
            assert ok
            m.assert_called_once()

    def test_load_safetensor_path(self):
        loader = _make_loader(safetensors=True)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        with patch.object(loader, "load_safetensor_fp8_from_disk", return_value=(True, "ok")):
            ok, _ = loader.load_experts_weight_from_disk()
            assert ok

    def test_load_disk_failure(self):
        loader = _make_loader()
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[0, 1], [2, 3]])
        with patch.object(loader, "load_weight_bf16_from_disk", return_value=(False, "disk error")):
            ok, msg = loader.load_experts_weight_from_disk()
            assert not ok

    def test_load_none_lists(self):
        loader = _make_loader()
        loader.old_model_ep_rank_to_expert_id_list = None
        loader.new_model_ep_rank_to_expert_id_list = None
        ok, _ = loader.load_experts_weight_from_disk()
        assert not ok

    def test_load_skips_before_start_index(self):
        loader = _make_loader(moe_layer_start_index=2)
        loader.old_model_ep_rank_to_expert_id_list = np.array([[0, 1], [0, 1], [0, 1]])
        loader.new_model_ep_rank_to_expert_id_list = np.array([[9, 9], [9, 9], [2, 3]])
        with patch.object(loader, "load_weight_bf16_from_disk", return_value=(True, "ok")) as m:
            loader.load_experts_weight_from_disk()
            layer_ids = [lid for lid, _ in m.call_args[0][0]]
            assert all(lid >= 2 for lid in layer_ids)

    def test_load_length_mismatch(self):
        loader = _make_loader()
        old = np.array([[0, 1], [0, 1]])
        new = np.empty(2, dtype=object)
        new[0] = np.array([0, 1])
        new[1] = np.array([0, 1, 2])
        loader.old_model_ep_rank_to_expert_id_list = old
        loader.new_model_ep_rank_to_expert_id_list = new
        ok, _ = loader.load_experts_weight_from_disk()
        assert not ok

    # load_weight_bf16_from_disk
    def test_bf16_missing_files(self, tmp_path):
        loader = _make_loader(model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
        with patch("paddle.device.get_device", return_value="cpu"), patch("paddle.set_device"):
            ok, _ = loader.load_weight_bf16_from_disk([(3, 0)])
            assert ok

    def test_bf16_exception(self, tmp_path):
        loader = _make_loader(model_dir=str(tmp_path))
        with patch("paddle.device.get_device", side_effect=RuntimeError("boom")):
            ok, msg = loader.load_weight_bf16_from_disk([(3, 0)])
            assert not ok and "boom" in msg

    def test_bf16_records_filenames(self, tmp_path):
        loader = _make_loader(model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
        with patch("paddle.device.get_device", return_value="cpu"), patch("paddle.set_device"):
            loader.load_weight_bf16_from_disk([(3, 0), (4, 1)])
            assert len(loader.moe_file_names) == 4

    # load_safetensor_fp8_from_disk
    def test_fp8_loads(self, tmp_path):
        loader = _make_loader(safetensors=True, model_dir=str(tmp_path), expert_per_rank=8, moe_layer_start_index=3)
        need = [(3, 0)]
        fake_map = {}
        names = []
        for proj in ["up_gate_proj", "down_proj"]:
            for quant in ["quant_weight", "weight_scale"]:
                n = f"ernie.layers.3.mlp.experts.0.{proj}.{quant}"
                fake_map[n] = str(tmp_path / "shard.safetensors")
                names.append(n)

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.keys.return_value = names
        mock_file.get_tensor.return_value = paddle.ones([4], dtype="float32")

        with (
            patch("fastdeploy.eplb.async_expert_loader.load_ep_checkpoint", return_value=fake_map),
            patch("safetensors.safe_open", return_value=mock_file),
            patch("paddle.device.get_device", return_value="cpu"),
            patch("paddle.set_device"),
        ):
            ok, _ = loader.load_safetensor_fp8_from_disk(need)
            assert ok and len(loader.cached_weights) == 4


# ── create_mmap + load_model_weights_process ───────────────────────────────
class TestMmapAndProcess:
    """create_mmap and load_model_weights_process."""

    def _mock_cudart(self, register_ok=True):
        m = MagicMock()

        class Err:
            cudaSuccess = 0
            cudaErrorInvalidValue = 1

        m.cudaError_t = Err
        ret = Err.cudaSuccess if register_ok else Err.cudaErrorInvalidValue
        m.cudaHostRegister.return_value = (ret,)
        m.cudaGetErrorString.return_value = (Err.cudaSuccess, b"err")
        return m

    def test_mmap_failure(self):
        with (
            patch("fastdeploy.eplb.async_expert_loader.cudart", self._mock_cudart()),
            patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
            patch("os.path.isfile", return_value=True),
            patch("os.open", return_value=5),
            patch("os.ftruncate"),
        ):
            ml.mmap.return_value = -1
            with pytest.raises(OSError):
                create_mmap(["m"], 0, 1, "u", _eplb_config())

    def test_cudart_none(self):
        with (
            patch("fastdeploy.eplb.async_expert_loader.cudart", None),
            patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
            patch("os.path.isfile", return_value=False),
            patch("builtins.open", MagicMock()),
            patch("os.open", return_value=5),
            patch("os.ftruncate"),
        ):
            ml.mmap.return_value = 12345
            with pytest.raises(ImportError):
                create_mmap(["m"], 0, 1, "u", _eplb_config())

    def test_cuda_register_failure(self):
        mc = self._mock_cudart(register_ok=False)
        with (
            patch("fastdeploy.eplb.async_expert_loader.cudart", mc),
            patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
            patch("os.path.isfile", return_value=False),
            patch("builtins.open", MagicMock()),
            patch("os.open", return_value=5),
            patch("os.ftruncate"),
            patch("ctypes.cast"),
            patch("ctypes.addressof", return_value=0x1000),
        ):
            ml.mmap.return_value = 12345
            with pytest.raises(RuntimeError):
                create_mmap(["m"], 0, 1, "u", _eplb_config())

    def test_default_shmem_size(self):
        cfg = _eplb_config(redundant_expert_async_load_model_shmem_size_gb=0)
        with (
            patch("fastdeploy.eplb.async_expert_loader.cudart", self._mock_cudart()),
            patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
            patch("os.path.isfile", return_value=False),
            patch("builtins.open", MagicMock()),
            patch("os.open", return_value=5),
            patch("os.ftruncate") as mt,
            patch("ctypes.cast"),
            patch("ctypes.addressof", return_value=0x1000),
        ):
            ml.mmap.return_value = 12345
            create_mmap(["m"], 0, 2, "u", cfg)
            mt.assert_called_once_with(5, 175 * 1024**3)

    def test_mmap_success_with_logger(self):
        logger = MagicMock()
        with (
            patch("fastdeploy.eplb.async_expert_loader.cudart", self._mock_cudart()),
            patch("fastdeploy.eplb.async_expert_loader.libc") as ml,
            patch("os.path.isfile", return_value=False),
            patch("builtins.open", MagicMock()),
            patch("os.open", return_value=5),
            patch("os.ftruncate"),
            patch("ctypes.cast"),
            patch("ctypes.addressof", return_value=0x1000),
        ):
            ml.mmap.return_value = 12345
            result = create_mmap(["m"], 0, 1, "u", _eplb_config(), logger)
            assert "m" in result

    # load_model_weights_process
    def _run_process(self, disk_ok=True, disk_exc=False):
        mg = MagicMock()
        data = MagicMock()
        mg.recv.side_effect = [
            {
                "old_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
                "new_model_ep_rank_to_expert_id_list": np.array([[0, 1]]),
            },
            KeyboardInterrupt,
        ]
        patches = [
            patch("setproctitle.setproctitle"),
            patch("faulthandler.enable"),
            patch("paddle.set_device"),
            patch("fastdeploy.utils.get_logger", return_value=MagicMock()),
        ]
        if disk_exc:
            patches.append(
                patch.object(AsyncEPLoader, "load_experts_weight_from_disk", side_effect=RuntimeError("boom"))
            )
        else:
            patches.append(
                patch.object(
                    AsyncEPLoader, "load_experts_weight_from_disk", return_value=(disk_ok, "ok" if disk_ok else "fail")
                )
            )
        if disk_ok and not disk_exc:
            patches.append(
                patch(
                    "fastdeploy.eplb.async_expert_loader.save_tensor_to_shm_mem",
                    return_value=[("w", 0, 4, [1], paddle.float32)],
                )
            )
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                load_model_weights_process(0, "/fake", 8, 3, "", "uuid", _eplb_config(), data, mg)
            except KeyboardInterrupt:
                pass
        return data

    def test_process_success(self):
        data = self._run_process(disk_ok=True)
        data.send.assert_called_once()
        assert data.send.call_args[0][0]["result"] is True

    def test_process_failure(self):
        data = self._run_process(disk_ok=False)
        data.send.assert_called_once()
        assert data.send.call_args[0][0]["result"] is False
